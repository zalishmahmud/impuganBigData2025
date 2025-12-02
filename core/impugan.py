from core.data_sampler import DataSampler
from core.data_transformer import DataTransformer
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm


class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        # print(f'inpu_.size() = {input_.size()} \n pac = {self.pac}' )
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)
        return data


class IMPUGAN():
    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        epochs=300,
        pac=10,
        cuda=True,
        # AGGRESSIVE PARAMETERS
        conditional_loss_weight=2.0,  # Much higher weight
        condition_loss_type='cross_entropy',  # New aggressive loss type value: exponential_penalty
        condition_temperature=1,  # Lower temperature for sharper distributions
        use_condition_aware_training=True,  # Focus training on multi-condition samples
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        # AGGRESSIVE PARAMETERS
        self._conditional_loss_weight = conditional_loss_weight
        self._condition_loss_type = condition_loss_type
        self._condition_temperature = condition_temperature
        self._use_condition_aware_training = use_condition_aware_training

        if torch.backends.mps.is_available():
          device = torch.device("mps")
          print(f"Using device: {device}")
        elif not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        print(f'Running model: {self.__class__.__name__}')
        print(f'AGGRESSIVE MODE: conditional_loss_weight={conditional_loss_weight}, '
              f'condition_loss_type={condition_loss_type}')

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed
        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    # Use lower temperature for sharper distributions
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=self._condition_temperature)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    def _compute_conditional_loss(self, data, c, m):
        return self._cross_entropy_conditional_loss(data, c, m)

    def _cross_entropy_conditional_loss(self, data, c, m):
        if m is None or m.sum() == 0:
            return torch.tensor(0.0, device=data.device), 0, 0

        total_loss = torch.tensor(0.0, device=data.device)
        total_satisfied = 0
        total_conditions = 0

        st_data = 0
        st_cond = 0
        col_idx = 0

        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) == 1 and span_info.activation_fn == 'softmax':
                    ed_data = st_data + span_info.dim
                    ed_cond = st_cond + span_info.dim

                    if col_idx < m.size(1):
                        column_mask = m[:, col_idx]
                        conditioned_samples = column_mask.sum()

                        if conditioned_samples > 0:
                            pred_logits = data[:, st_data:ed_data]
                            true_labels = torch.argmax(c[:, st_cond:ed_cond], dim=1)
                            ce_loss = functional.cross_entropy(pred_logits, true_labels, reduction='none')
                            masked_loss = ce_loss * column_mask
                            total_loss += masked_loss.sum()
                            total_conditions += conditioned_samples.item()

                            satisfied = (ce_loss < 0.5) * column_mask
                            total_satisfied += satisfied.sum().item()

                    st_cond = ed_cond
                    col_idx += 1

                st_data += span_info.dim

        if total_conditions > 0:
            return total_loss / total_conditions, total_satisfied, total_conditions
        else:
            return torch.tensor(0.0, device=data.device), 0, 0

    def _validate_discrete_columns(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def fit(self, train_data, discrete_columns=(), epochs=None):
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs

        print(f"IMPUGAN Training:")
        print(f"  - Conditional loss type: {self._condition_loss_type}")
        print(f"  - Conditional loss weight: {self._conditional_loss_weight}")

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency, transformer= self._transformer
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=[
            'Epoch', 'Generator Loss', 'Discriminator Loss', 'Conditional Loss',
            'Condition Accuracy'
        ])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen: {gen:.3f} | Disc: {dis:.3f} | Cond: {cond:.3f} | Acc: {acc:.1f}%'
            epoch_iterator.set_description(description.format(gen=0, dis=0, cond=0, acc=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        for i in epoch_iterator:
            epoch_cond_loss = 0
            epoch_accuracy = 0
            epoch_steps = 0

            for id_ in range(steps_per_epoch):
                # Train discriminator
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    # FORCE multi-condition sampling more often
                    if self._use_condition_aware_training:
                        # 70% chance of multi-condition, 30% single condition
                        if np.random.random() < 0.7:
                            condvec = self._data_sampler.sample_condvec_multi(self._batch_size, num_conditions=2)
                        else:
                            condvec = self._data_sampler.sample_condvec_multi(self._batch_size, num_conditions=1)
                    else:
                        condvec = self._data_sampler.sample_condvec_multi(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data_multi_cond_improved(
                            train_data, self._batch_size, [], []
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data_multi_cond_improved(
                            train_data, self._batch_size, col, opt
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # Train generator with AGGRESSIVE conditioning
                fakez = torch.normal(mean=mean, std=std)

                # FORCE multi-condition sampling for generator training
                if self._use_condition_aware_training:
                    condvec = self._data_sampler.sample_condvec_multi(self._batch_size)
                else:
                    condvec = self._data_sampler.sample_condvec_multi(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    conditional_loss = torch.tensor(0.0, device=self._device)
                    satisfied = 0
                    total_cond = 0
                else:
                    conditional_loss, satisfied, total_cond = self._compute_conditional_loss(fake, c1, m1)

                # MASSIVE conditional loss weight
                adversarial_loss = -torch.mean(y_fake)
                weighted_conditional_loss = conditional_loss * self._conditional_loss_weight
                loss_g = adversarial_loss + weighted_conditional_loss

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                # Track metrics
                epoch_cond_loss += conditional_loss.item()
                if total_cond > 0:
                    epoch_accuracy += (satisfied / total_cond) * 100
                epoch_steps += 1

            # Log epoch results
            avg_cond_loss = epoch_cond_loss / max(epoch_steps, 1)
            avg_accuracy = epoch_accuracy / max(epoch_steps, 1)


            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Conditional Loss': [avg_cond_loss],
                'Condition Accuracy': [avg_accuracy]
            })

            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(
                        gen=generator_loss,
                        dis=discriminator_loss,
                        cond=avg_cond_loss,
                        acc=avg_accuracy
                    )
                )

    def sample_with_guarantee(self, n, conditions, max_attempts=100, batch_size=None):
        if batch_size is None:
            batch_size = max(self._batch_size, 100)  # Use larger batch for better success rate

        print(f"Sampling {n} samples with guaranteed conditions:")
        for cond in conditions:
            print(f"  - {cond['column']} = {cond['value']}")

        # Check feasibility first
        feasible_count, feasible_percent = self._data_sampler.check_multi_condition_feasibility(
            conditions, self._transformer
        )

        if feasible_percent < 0.1:
            print(f"WARNING: Very low feasibility ({feasible_percent:.3f}%). This may take a long time.")

        results = []
        attempts = 0
        total_samples_generated = 0

        while len(results) < n and attempts < max_attempts:
            # Sample a batch using improved sampling
            batch = self.sample(batch_size, conditions)
            total_samples_generated += batch_size

            # Check which samples satisfy ALL conditions
            mask = pd.Series([True] * len(batch))
            for cond in conditions:
                col = cond['column']
                val = cond['value']
                mask = mask & (batch[col] == val)

            # Add satisfying samples
            satisfying = batch[mask]
            if len(satisfying) > 0:
                results.append(satisfying)
                print(f"Attempt {attempts+1}: Found {len(satisfying)} samples "
                      f"({len(satisfying)/batch_size*100:.1f}% success rate)")

            attempts += 1

            # Progress info
            current_count = sum(len(df) for df in results)
            if attempts % 20 == 0:
                overall_success = current_count / total_samples_generated * 100
                print(f"Progress: {current_count}/{n} samples found "
                      f"(overall success rate: {overall_success:.2f}%)")

        if results:
            # Combine all satisfying results
            all_satisfying = pd.concat(results, ignore_index=True)
            final_results = all_satisfying.head(n)

            # Compute success rate
            success_rate = len(all_satisfying) / total_samples_generated * 100

            print(f"\nSUCCESS! Found {len(final_results)}/{n} samples in {attempts} attempts")
            print(f"Overall success rate: {success_rate:.2f}% "
                  f"({len(all_satisfying)} satisfying out of {total_samples_generated} generated)")
            print(f"Expected vs actual: {feasible_percent:.2f}% vs {success_rate:.2f}%")

            return final_results
        else:
            print(f"\nFAILED: Could not find {n} satisfying samples in {max_attempts} attempts")
            print(f"Generated {total_samples_generated} total samples")
            return pd.DataFrame()

    def sample(self, n, conditions=None):
        if conditions is not None:
            # Generate condition vector and mask
            global_condition_vec, global_mask = self._data_sampler.generate_multi_cond_from_conditions(
                conditions, self._batch_size, self._transformer
            )
        else:
            global_condition_vec = None
            global_mask = None

        effective_batch_size = self._batch_size
        steps = max(n // effective_batch_size + 1, 3)  # At least 3 attempts
        data = []

        for i in range(steps):
            mean = torch.zeros(effective_batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                # Use the specific conditions provided
                condvec = global_condition_vec.copy()
                c1 = torch.from_numpy(condvec).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            else:
                # Use random conditions
                condvec = self._data_sampler.sample_condvec_multi(effective_batch_size)
                if condvec is not None:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def evaluate_condition_satisfaction(self, samples, conditions):
        if samples.empty or not conditions:
            return {"total_samples": 0, "satisfied_samples": 0, "satisfaction_rate": 0.0}

        # Check which samples satisfy ALL conditions
        mask = pd.Series([True] * len(samples))
        condition_details = {}

        for cond in conditions:
            col = cond['column']
            val = cond['value']

            if col in samples.columns:
                col_mask = samples[col] == val
                satisfied_count = col_mask.sum()
                condition_details[f"{col}={val}"] = {
                    "satisfied": satisfied_count,
                    "total": len(samples),
                    "rate": satisfied_count / len(samples) * 100
                }
                mask = mask & col_mask
            else:
                print(f"WARNING: Column '{col}' not found in samples")
                condition_details[f"{col}={val}"] = {
                    "satisfied": 0,
                    "total": len(samples),
                    "rate": 0.0
                }
                mask = pd.Series([False] * len(samples))

        total_satisfied = mask.sum()
        satisfaction_rate = total_satisfied / len(samples) * 100

        result = {
            "total_samples": len(samples),
            "satisfied_samples": total_satisfied,
            "satisfaction_rate": satisfaction_rate,
            "condition_details": condition_details
        }

        return result

    def get_training_summary(self):
        if self.loss_values is None or self.loss_values.empty:
            return "No training data available"

        recent_epochs = self.loss_values.tail(10)

        summary = "IMPUGAN Training Summary:\n"
        summary += "=" * 50 + "\n"
        summary += f"Total Epochs Trained: {len(self.loss_values)}\n"
        summary += f"Configuration:\n"
        summary += f"  - Conditional Loss Type: {self._condition_loss_type}\n"
        summary += f"  - Conditional Loss Weight: {self._conditional_loss_weight}\n"
        summary += f"  - Batch Size: {self._batch_size}\n"
        summary += f"  - Generator LR: {self._generator_lr}\n"
        summary += f"  - Discriminator LR: {self._discriminator_lr}\n\n"

        summary += "Recent Performance (last 10 epochs):\n"
        summary += f"  - Avg Generator Loss: {recent_epochs['Generator Loss'].mean():.4f}\n"
        summary += f"  - Avg Discriminator Loss: {recent_epochs['Discriminator Loss'].mean():.4f}\n"
        summary += f"  - Avg Conditional Loss: {recent_epochs['Conditional Loss'].mean():.4f}\n"
        summary += f"  - Avg Condition Accuracy: {recent_epochs['Condition Accuracy'].mean():.1f}%\n"
        summary += f"  - Multi-condition Training Ratio: {recent_epochs['Multi_Condition_Samples'].mean():.1f}\n\n"

        # Loss trends
        if len(self.loss_values) > 20:
            early_epochs = self.loss_values.head(10)

            gen_trend = recent_epochs['Generator Loss'].mean() - early_epochs['Generator Loss'].mean()
            cond_trend = recent_epochs['Conditional Loss'].mean() - early_epochs['Conditional Loss'].mean()
            acc_trend = recent_epochs['Condition Accuracy'].mean() - early_epochs['Condition Accuracy'].mean()

            summary += "Training Trends (recent vs early):\n"
            summary += f"  - Generator Loss: {'↓' if gen_trend < 0 else '↑'} {abs(gen_trend):.4f}\n"
            summary += f"  - Conditional Loss: {'↓' if cond_trend < 0 else '↑'} {abs(cond_trend):.4f}\n"
            summary += f"  - Condition Accuracy: {'↑' if acc_trend > 0 else '↓'} {abs(acc_trend):.1f}%\n"

        return summary

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def _apply_hard_conditions_to_generated(self, fake, cond, mask):
        """Force generator outputs to match requested *discrete* conditions.

        Only override softmax spans that correspond to DISCRETE columns
        (i.e., columns whose output_info_list entries have length == 1 and softmax).
        Softmax spans that belong to CONTINUOUS columns (the GMM component) are left untouched.
        """
        if cond is None or mask is None:
            return fake

        st_data = 0
        st_cond = 0
        col_idx = 0
        pieces = []

        for column_info in self._transformer.output_info_list:
            is_discrete_col = (len(column_info) == 1 and column_info[0].activation_fn == 'softmax')
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st_data + span_info.dim
                    pieces.append(fake[:, st_data:ed])
                    st_data = ed
                elif span_info.activation_fn == 'softmax':
                    ed_data = st_data + span_info.dim

                    if is_discrete_col:
                        # true discrete column; align with cond/mask
                        ed_cond = st_cond + span_info.dim
                        col_mask = mask[:, col_idx:col_idx + 1]  # (B, 1)
                        target = cond[:, st_cond:ed_cond]  # (B, K)
                        current = fake[:, st_data:ed_data]  # (B, K)
                        overridden = target * col_mask + current * (1 - col_mask)
                        pieces.append(overridden)
                        st_cond = ed_cond
                        col_idx += 1
                    else:
                        # softmax for continuous component — leave unchanged
                        pieces.append(fake[:, st_data:ed_data])

                    st_data = ed_data
                else:
                    raise ValueError(f"Unexpected activation {span_info.activation_fn}.")

        return torch.cat(pieces, dim=1)

    def sample_hard_conditions(self, n, conditions, batch_size=None, continuous_columns=None, discrete_columns=None):
        """Generate samples that *guarantee* discrete conditions by hard-overriding softmax segments.

        This is rejection-free and typically yields ~100% satisfaction for discrete columns.
        """
        if batch_size is None:
            batch_size = max(512, self._batch_size)

        # Build a fixed conditional vector/mask for the user-specified conditions
        cond, mask, cols, opts = self._data_sampler.build_condvec_from_user_conditions(
            self._transformer, conditions, batch_size
        )

        steps = (n + batch_size - 1) // batch_size
        out_frames = []
        self._generator.eval()

        with torch.no_grad():
            for _ in range(steps):
                mean = torch.zeros(batch_size, self._embedding_dim, device=self._device)
                std = mean + 1
                z = torch.normal(mean=mean, std=std)

                if cond is not None:
                    c = torch.from_numpy(cond).to(self._device)
                    m = torch.from_numpy(mask).to(self._device)
                    inp = torch.cat([z, c], dim=1)
                else:
                    c = None
                    m = None
                    inp = z

                raw = self._generator(inp)
                act = self._apply_activate(raw)
                act = self._apply_hard_conditions_to_generated(act, c, m)

                # Inverse transform expects numpy array
                df = self._transformer.inverse_transform(act.detach().cpu().numpy())
                out_frames.append(df)

        result_df = pd.concat(out_frames, ignore_index=True).iloc[:n]
        
        return result_df