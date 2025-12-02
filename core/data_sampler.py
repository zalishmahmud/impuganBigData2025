"""DataSampler module."""

import numpy as np


class DataSampler(object):
    def __init__(self, data, output_info, log_frequency, transformer=None):
        self._data_length = len(data)
        self.transformer = transformer

        def is_discrete_column(column_info):
            return len(column_info) == 1 and column_info[0].activation_fn == 'softmax'

        n_discrete_columns = sum([
            1 for column_info in output_info if is_discrete_column(column_info)
        ])

        # Store the row id for each category in each discrete column
        self._rid_by_cat_cols = []

        # Build category mappings
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

        # Prepare conditional vector matrices
        max_category = max(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim for column_info in output_info if is_discrete_column(column_info)
        ])

        # Calculate probabilities and positions
        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

        # Add debugging info
        print(f"DataSampler initialized:")
        print(f"  - Total discrete columns: {self._n_discrete_columns}")
        print(f"  - Total categories: {self._n_categories}")
        print(f"  - Data length: {self._data_length}")

        # Print category distribution for debugging
        for i in range(self._n_discrete_columns):
            n_cats = self._discrete_column_n_category[i]
            print(f"  - Column {i}: {n_cats} categories")

            # Try to get the actual category labels from transformer
            category_labels = None
            if hasattr(self, "transformer") and self.transformer is not None:
                try:
                    # Filter out discrete column infos only
                    discrete_infos = [
                        info for info in self.transformer._column_transform_info_list
                        if info.column_type == "discrete"
                    ]
                    if i < len(discrete_infos):
                        category_labels = discrete_infos[i].transform.dummies
                except Exception as e:
                    print(f"      [!] Could not retrieve labels for column {i}: {e}")

            for j in range(n_cats):
                count = len(self._rid_by_cat_cols[i][j])
                label = category_labels[j] if category_labels is not None and j < len(category_labels) else f"cat_{j}"
                print(f"    Category {j} ({label}): {count} rows ({count / self._data_length * 100:.1f}%)")

    def _random_choice_prob_index(self, discrete_column_id):
        """Sample category index based on probabilities."""
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.random.rand()
        cumsum = np.cumsum(probs[:self._discrete_column_n_category[discrete_column_id]])
        return np.searchsorted(cumsum, r)

    def sample_condvec_multi(self, batch, num_conditions=None):
        """Generate conditional vector with flexible number of conditions."""
        if self._n_discrete_columns == 0:
            return None

        if num_conditions is None:
            # Bias towards multiple conditions for better multi-condition training
            max_possible = int(self._n_discrete_columns * (1 / 3))
            num_conditions = np.random.choice([1, 2, max_possible ], p=[0.3, 0.5, 0.2])
            num_conditions = min(num_conditions, self._n_discrete_columns)

        # Ensure we don't exceed available columns
        num_conditions = min(num_conditions, self._n_discrete_columns)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        discrete_column_ids = []
        category_ids_in_col = []

        for i in range(batch):
            # Randomly select which columns to condition on
            selected_columns = np.random.choice(
                self._n_discrete_columns,
                size=num_conditions,
                replace=False
            )

            batch_discrete_cols = []
            batch_category_ids = []

            for col_id in selected_columns:
                # Sample category for this column
                category_id_in_col = self._random_choice_prob_index(col_id)

                # Calculate global category index
                category_id = self._discrete_column_cond_st[col_id] + category_id_in_col

                # Set the condition
                cond[i, category_id] = 1
                mask[i, col_id] = 1

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(category_id_in_col)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        return cond, mask, discrete_column_ids, category_ids_in_col

    def sample_data_multi_cond_improved(self, data, n, col_lists, opt_lists):
        """Improved multi-condition data sampling with better intersection strategy."""
        if col_lists is None or all(len(cols) == 0 for cols in col_lists):
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        intersection_failures = 0
        fallback_usage = 0

        for cols, opts in zip(col_lists, opt_lists):
            if len(cols) == 0:
                idx.append(np.random.randint(len(data)))
            else:
                # Find rows that satisfy ALL conditions
                valid_rows = None
                condition_sets = []

                for c, o in zip(cols, opts):
                    rows_for_condition = set(self._rid_by_cat_cols[c][o])
                    condition_sets.append(rows_for_condition)
                    if valid_rows is None:
                        valid_rows = rows_for_condition
                    else:
                        valid_rows = valid_rows.intersection(rows_for_condition)

                if len(valid_rows) > 0:
                    # Success: found rows satisfying all conditions
                    idx.append(np.random.choice(list(valid_rows)))
                else:
                    intersection_failures += 1
                    # Progressive fallback strategy

                    # 1. Try pairs of conditions
                    best_intersection = set()
                    for i in range(len(condition_sets)):
                        for j in range(i+1, len(condition_sets)):
                            pair_intersection = condition_sets[i].intersection(condition_sets[j])
                            if len(pair_intersection) > len(best_intersection):
                                best_intersection = pair_intersection

                    if len(best_intersection) > 0:
                        idx.append(np.random.choice(list(best_intersection)))
                        fallback_usage += 1
                    else:
                        # 2. Use largest single condition set
                        largest_set = max(condition_sets, key=len)
                        if len(largest_set) > 0:
                            idx.append(np.random.choice(list(largest_set)))
                            fallback_usage += 1
                        else:
                            # 3. Random fallback
                            idx.append(np.random.randint(len(data)))
                            fallback_usage += 1

        # Print debugging info occasionally
        # if intersection_failures > 0:
        #     print(f"Multi-condition sampling: {intersection_failures}/{n} intersection failures, "
        #           f"{fallback_usage} fallback usages")

        return data[idx]

    def sample_condvec_from_real_data(self, batch, data=None, debug=False):
        """
        Ensure every discrete column is used — each row conditions on *all* discrete columns.
        FOR DEBUGGING: Always uses the same fixed rows (0, 1, 2, ...)
        """
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        discrete_column_ids = []
        category_ids_in_col = []

        # FIXED SAMPLING FOR DEBUGGING: Always use rows 0, 1, 2, 3, ...
        chosen_rows = np.arange(batch) % len(data)

        if debug:
            print(f"\n{'=' * 70}")
            print(f"🔍 DEBUG MODE: Using FIXED rows: {chosen_rows}")
            print(f"{'=' * 70}")

        # Get discrete column information from transformer
        discrete_infos = []
        if hasattr(self, "transformer") and self.transformer is not None:
            try:
                discrete_infos = [
                    info for info in self.transformer._column_transform_info_list
                    if info.column_type == "discrete"
                ]
            except Exception as e:
                if debug:
                    print(f"[!] Could not retrieve transformer info: {e}")

        # Build discrete spans
        col_spans = []
        st = 0
        discrete_col_counter = 0
        column_names = []

        for column_info in self.transformer._column_transform_info_list:
            if column_info.column_type == 'discrete':
                dim = column_info.output_dimensions
                col_spans.append((discrete_col_counter, st, st + dim))
                column_names.append(column_info.column_name)
                discrete_col_counter += 1
            st += column_info.output_dimensions

        if debug:
            print(f"\n📊 Dataset Info:")
            print(f"   Discrete Columns: {column_names}")
            print(f"   Number of discrete columns: {len(discrete_infos)}")
            print(f"   Total categories: {self._n_categories}")

        # Process each sample
        for i, row_id in enumerate(chosen_rows):
            batch_discrete_cols, batch_category_ids = [], []

            for col_idx, (col_id, st, ed) in enumerate(col_spans):
                # Extract one-hot segment
                onehot_segment = data[row_id, st:ed]
                category_idx = np.argmax(onehot_segment)

                # ⚠️ BUG FIX: Calculate correct global position
                # OLD WRONG: global_id = self._discrete_column_cond_st[0] + 0
                # NEW CORRECT:
                global_id = self._discrete_column_cond_st[col_id] + category_idx

                # Set condition and mask
                cond[i, global_id] = 1.0
                mask[i, col_id] = 1.0

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(category_idx)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        # PRINT SUMMARY OF WHAT WAS CHOSEN
        if debug:
            print(f"\n{'=' * 70}")
            print(f"📋 SAMPLES CHOSEN (always the same for debugging):")
            print(f"{'=' * 70}\n")

            for i, row_id in enumerate(chosen_rows):
                print(f"  Sample {i} (from Row {row_id}):")

                for col_idx in range(len(column_names)):
                    cat_idx = category_ids_in_col[i][col_idx]
                    human_val = "?"

                    # Get human-readable value
                    if col_idx < len(discrete_infos):
                        try:
                            labels = discrete_infos[col_idx].transform.dummies
                            if cat_idx < len(labels):
                                human_val = labels[cat_idx]
                        except Exception as e:
                            human_val = f"error: {e}"

                    print(f"    • {column_names[col_idx]:20} = '{human_val}' (index: {cat_idx})")

                print()

            print(f"{'=' * 70}\n")

        return cond, mask, discrete_column_ids, category_ids_in_col
    def sample_condvec_from_real_data_old(self,batch, data=None, num_conditions=None):
        """
        Sample condition vectors based on actual co-occurring discrete values from the given data.
        Uses real discrete column combinations from the dataset instead of random category sampling.
        """
        if self._n_discrete_columns == 0:
            return None

        if num_conditions is None:
            max_possible = max(1, int(self._n_discrete_columns * (1 / 3)))
            num_conditions = np.random.choice([1, 2, max_possible], p=[0.3, 0.5, 0.2])
            num_conditions = min(num_conditions, self._n_discrete_columns)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        discrete_column_ids = []
        category_ids_in_col = []

        # === Randomly pick real data rows ===
        chosen_rows = np.random.choice(len(data), batch, replace=True)

        # === Get spans for discrete columns (from transformer) ===
        col_spans = []
        st = 0
        discrete_col_counter = 0
        for column_info in self.transformer._column_transform_info_list:
            if column_info.column_type == 'discrete':
                dim = column_info.output_dimensions
                col_spans.append((discrete_col_counter, st, st + dim))
                discrete_col_counter += 1
            st += column_info.output_dimensions

        # === Iterate through sampled rows ===
        for i, row_id in enumerate(chosen_rows):
            selected_cols = np.random.choice(len(col_spans), size=num_conditions, replace=False)
            batch_discrete_cols, batch_category_ids = [], []

            for c in selected_cols:
                col_id, st, ed = col_spans[c]
                onehot = np.argmax(data[row_id, st:ed])
                global_id = self._discrete_column_cond_st[col_id] + onehot

                cond[i, global_id] = 1.0
                mask[i, col_id] = 1.0

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(onehot)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        return cond, mask, discrete_column_ids, category_ids_in_col

    def generate_multi_cond_from_conditions(self, conditions, batch, transformer):
        """Generate conditional vector from multiple column-value pairs."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        # print(f"Generating condition vector for {len(conditions)} conditions:")

        for condition in conditions:
            # print(f"  - {condition['column']} = {condition['value']}")

            # Convert column name and value to indices
            try:
                condition_info = transformer.convert_column_name_value_to_id(
                    condition['column'], condition['value']
                )

                # Set the corresponding position to 1
                col_id = condition_info['discrete_column_id']
                value_id = condition_info['value_id']
                global_id = self._discrete_column_cond_st[col_id] + value_id

                vec[:, global_id] = 1
                mask[:, col_id] = 1

                # print(f"    Mapped to: col_id={col_id}, value_id={value_id}, global_id={global_id}")

                # Check how many training samples satisfy this condition
                # matching_rows = len(self._rid_by_cat_cols[col_id][value_id])
                # print(f"    Training samples with this condition: {matching_rows}/{self._data_length} "
                #       f"({matching_rows/self._data_length*100:.1f}%)")

            except Exception as e:
                print(f"    ERROR mapping condition: {e}")

        return vec, mask

    def check_multi_condition_feasibility(self, conditions, transformer):
        """Check how many training samples satisfy all given conditions."""
        if not conditions:
            return self._data_length, 100.0

        valid_rows = None

        print(f"Checking feasibility for {len(conditions)} conditions:")

        for condition in conditions:
            try:
                condition_info = transformer.convert_column_name_value_to_id(
                    condition['column'], condition['value']
                )

                col_id = condition_info['discrete_column_id']
                value_id = condition_info['value_id']

                rows_for_condition = set(self._rid_by_cat_cols[col_id][value_id])
                print(f"  - {condition['column']} = {condition['value']}: {len(rows_for_condition)} samples")

                if valid_rows is None:
                    valid_rows = rows_for_condition
                else:
                    valid_rows = valid_rows.intersection(rows_for_condition)
                    print(f"    After intersection: {len(valid_rows)} samples")

            except Exception as e:
                print(f"  - ERROR with condition {condition}: {e}")
                return 0, 0.0

        if valid_rows is None:
            return 0, 0.0

        count = len(valid_rows)
        percentage = count / self._data_length * 100

        print(f"Total samples satisfying ALL conditions: {count}/{self._data_length} ({percentage:.2f}%)")

        return count, percentage

    def sample_original_condvec(self, batch):
        """Original single-condition sampling for compatibility."""
        if self._n_discrete_columns == 0:
            return None

        return self.sample_condvec_multi(batch, num_conditions=1)[0]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def build_condvec_from_user_conditions(self, transformer, conditions, batch, debug=False):
        """Build (cond, mask) from [{'column': <name>, 'value': <value>}, ...] replicated to batch."""
        if not conditions:
            return None, None, [], []

        if debug:
            print(f"\n{'─' * 80}")
            print(f"build_condvec_from_user_conditions called:")
            print(f"{'─' * 80}")
            print(f"  Batch size: {batch}")
            print(f"  Number of conditions: {len(conditions)}")
            print(f"  Total categories: {self._n_categories}")
            print(f"  Number of discrete columns: {self._n_discrete_columns}")

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        cols, opts = [], []

        for idx, condition in enumerate(conditions):
            if debug:
                print(f"\n  Processing condition [{idx}]: {condition['column']} = {condition['value']}")

            try:
                info = transformer.convert_column_name_value_to_id(
                    condition['column'],
                    condition['value']
                )

                col_id = info['discrete_column_id']  # which discrete column
                val_id_in_col = info['value_id']  # index within that column
                # Compute global category slot for the one-hot
                global_cat_id = self._discrete_column_cond_st[col_id] + val_id_in_col

                if debug:
                    print(f"    Mapping result:")
                    print(f"      discrete_column_id: {col_id}")
                    print(f"      value_id (in column): {val_id_in_col}")
                    print(f"      _discrete_column_cond_st[{col_id}]: {self._discrete_column_cond_st[col_id]}")
                    print(f"      global_cat_id: {global_cat_id}")

                cond[:, global_cat_id] = 1.0
                mask[:, col_id] = 1.0
                cols.append(col_id)
                opts.append(val_id_in_col)

                if debug:
                    print(f"    Set cond[:, {global_cat_id}] = 1.0")
                    print(f"    Set mask[:, {col_id}] = 1.0")

            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                raise

        if debug:
            print(f"\n  Summary:")
            print(f"    cond shape: {cond.shape}")
            print(f"    mask shape: {mask.shape}")
            print(f"    cond[0] non-zero indices: {np.where(cond[0] > 0)[0]}")
            print(f"    cond[0] non-zero values: {cond[0][cond[0] > 0]}")
            print(f"    mask[0] non-zero indices: {np.where(mask[0] > 0)[0]}")
            print(f"    mask[0] non-zero values: {mask[0][mask[0] > 0]}")
            print(f"    cols: {cols}")
            print(f"    opts: {opts}")

            # Verify all rows have same condition
            all_same = True
            for i in range(1, min(batch, 3)):
                if not np.array_equal(cond[0], cond[i]):
                    all_same = False
                    print(f"    ⚠️ WARNING: cond[0] != cond[{i}]")
                    break
            if all_same:
                print(f"    ✓ All batch rows have identical conditions")

        return cond, mask, cols, opts


    # Add to DataSampler class in data_sampler.py

    def sample_condvec_flexible(self, batch, strategy='weighted', epoch=None, total_epochs=None):
        """
        Flexible conditioning that supports 1 to N column combinations.

        Args:
            batch: Batch size
            strategy: 'uniform', 'weighted', or 'curriculum'
            epoch: Current epoch (for curriculum)
            total_epochs: Total epochs (for curriculum)
        """
        if self._n_discrete_columns == 0:
            return None

        if strategy == 'uniform':
            return self._sample_uniform(batch)
        elif strategy == 'weighted':
            return self._sample_weighted(batch)
        elif strategy == 'curriculum':
            if epoch is None or total_epochs is None:
                raise ValueError("epoch and total_epochs required for curriculum strategy")
            return self._sample_curriculum(batch, epoch, total_epochs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def _sample_uniform(self, batch):
        """Uniform distribution over all combination sizes."""
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        discrete_column_ids = []
        category_ids_in_col = []

        for i in range(batch):
            # Uniform choice of number of conditions
            num_conditions = np.random.randint(1, self._n_discrete_columns + 1)

            selected_columns = np.random.choice(
                self._n_discrete_columns,
                size=num_conditions,
                replace=False
            )

            batch_discrete_cols = []
            batch_category_ids = []

            for col_id in selected_columns:
                category_id_in_col = self._random_choice_prob_index(col_id)
                category_id = self._discrete_column_cond_st[col_id] + category_id_in_col

                cond[i, category_id] = 1
                mask[i, col_id] = 1

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(category_id_in_col)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        return cond, mask, discrete_column_ids, category_ids_in_col


    def _sample_weighted(self, batch):
        """Weighted distribution favoring fewer conditions."""
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        discrete_column_ids = []
        category_ids_in_col = []

        # Exponential decay weights
        weights = np.array([1.0 / (i + 1) for i in range(self._n_discrete_columns)])
        weights = weights / weights.sum()

        for i in range(batch):
            num_conditions = np.random.choice(
                range(1, self._n_discrete_columns + 1),
                p=weights
            )

            selected_columns = np.random.choice(
                self._n_discrete_columns,
                size=num_conditions,
                replace=False
            )

            batch_discrete_cols = []
            batch_category_ids = []

            for col_id in selected_columns:
                category_id_in_col = self._random_choice_prob_index(col_id)
                category_id = self._discrete_column_cond_st[col_id] + category_id_in_col

                cond[i, category_id] = 1
                mask[i, col_id] = 1

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(category_id_in_col)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        return cond, mask, discrete_column_ids, category_ids_in_col


    def _sample_curriculum(self, batch, epoch, total_epochs):
        """Curriculum learning: gradually increase complexity."""
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        discrete_column_ids = []
        category_ids_in_col = []

        progress = epoch / total_epochs

        # Determine max conditions based on progress
        if progress < 0.3:
            max_conditions = min(3, self._n_discrete_columns)
            weights = [0.5, 0.35, 0.15][:max_conditions]
        elif progress < 0.6:
            max_conditions = min(6, self._n_discrete_columns)
            weights = [0.3, 0.25, 0.2, 0.12, 0.08, 0.05][:max_conditions]
        else:
            max_conditions = self._n_discrete_columns
            weights = [1.0 / (i + 1) for i in range(max_conditions)]

        weights = np.array(weights)
        weights = weights / weights.sum()

        for i in range(batch):
            num_conditions = np.random.choice(
                range(1, max_conditions + 1),
                p=weights
            )

            selected_columns = np.random.choice(
                self._n_discrete_columns,
                size=num_conditions,
                replace=False
            )

            batch_discrete_cols = []
            batch_category_ids = []

            for col_id in selected_columns:
                category_id_in_col = self._random_choice_prob_index(col_id)
                category_id = self._discrete_column_cond_st[col_id] + category_id_in_col

                cond[i, category_id] = 1
                mask[i, col_id] = 1

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(category_id_in_col)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        return cond, mask, discrete_column_ids, category_ids_in_col

    def sample_condvec_flexible_from_real_data(self, batch, data, strategy='weighted', epoch=None, total_epochs=None,
                                               rare_boost=0.5):
        """
        Best of both worlds with RARE CLASS OVERSAMPLING:
        - Flexible number of conditions (1-N columns)
        - Conditions come from REAL data rows (guaranteed to exist)
        - HEAVILY oversample rare class values

        Args:
            batch: Batch size
            data: Transformed training data
            strategy: 'uniform', 'weighted', or 'curriculum'
            epoch: Current epoch (for curriculum)
            total_epochs: Total epochs (for curriculum)
            rare_boost: Probability of forcing rare class selection (0.0-1.0)
        """
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        discrete_column_ids = []
        category_ids_in_col = []

        # Define rare classes (those with < 20% frequency)
        rare_classes = {
            0: [1],  # sex: Female (30.8%)
            1: [1, 2],  # dataset: Hungary, VA Long Beach (0.4% each)
            2: [0, 3],  # cp: atypical angina (15.8%), typical angina (8.1%)
            3: [1],  # fbs: True (16.7%)
            4: [1],  # restecg: st-t abnormality (0.9%)
            6: [2],  # slope: downsloping (7.3%)
            7: [2, 3],  # ca: 2.0 (12.8%), 3.0 (7.3%)
            8: [2],  # thal: fixed defect (5.6%)
            9: [0, 2, 3, 4],  # num: 1 (18.4%), 3 (12.8%), 2 (12.4%), 4 (4.3%)
        }

        # Determine weights based on strategy
        if strategy == 'uniform':
            weights = np.ones(self._n_discrete_columns) / self._n_discrete_columns
        elif strategy == 'weighted':
            # 60% single condition, exponential decay for rest
            single_cond_weight = 0.60
            remaining_weight = 1.0 - single_cond_weight
            decay_weights = np.array([2.0 ** (-i) for i in range(self._n_discrete_columns - 1)])
            decay_weights = decay_weights / decay_weights.sum() * remaining_weight
            weights = np.concatenate([[single_cond_weight], decay_weights])
        elif strategy == 'curriculum':
            if epoch is None or total_epochs is None:
                raise ValueError("epoch and total_epochs required for curriculum")
            progress = epoch / total_epochs

            if progress < 0.3:
                weights = np.array([0.7, 0.3] + [0.0] * (self._n_discrete_columns - 2))
            elif progress < 0.6:
                weights = np.array([0.5, 0.25, 0.15, 0.1] + [0.0] * (self._n_discrete_columns - 4))
            else:
                weights = np.array([0.4] + [0.6 / (self._n_discrete_columns - 1)] * (self._n_discrete_columns - 1))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Get discrete column spans
        col_spans = []
        st = 0
        discrete_col_counter = 0
        for column_info in self.transformer._column_transform_info_list:
            if column_info.column_type == 'discrete':
                dim = column_info.output_dimensions
                col_spans.append((discrete_col_counter, st, st + dim))
                discrete_col_counter += 1
            st += column_info.output_dimensions

        for i in range(batch):
            # 🔥 RARE CLASS OVERSAMPLING
            if np.random.random() < rare_boost:
                # Force selection of a row with at least one rare class
                rare_row_candidates = set()

                for col_id, rare_cat_indices in rare_classes.items():
                    for cat_idx in rare_cat_indices:
                        if col_id < len(self._rid_by_cat_cols) and cat_idx < len(self._rid_by_cat_cols[col_id]):
                            rare_row_candidates.update(self._rid_by_cat_cols[col_id][cat_idx])

                if rare_row_candidates:
                    row_id = np.random.choice(list(rare_row_candidates))
                else:
                    row_id = np.random.choice(len(data))
            else:
                # Normal random sampling
                row_id = np.random.choice(len(data))

            # Decide how many conditions to use for this sample
            num_conditions = np.random.choice(
                range(1, self._n_discrete_columns + 1),
                p=weights
            )

            # 🔥 PRIORITIZE RARE CLASS COLUMNS
            # If we sampled a rare row, make sure to include the rare columns
            if np.random.random() < rare_boost:
                # Find which columns in this row have rare values
                rare_cols_in_row = []
                for col_idx, (col_id, st, ed) in enumerate(col_spans):
                    onehot_segment = data[row_id, st:ed]
                    category_idx = np.argmax(onehot_segment)

                    if col_id in rare_classes and category_idx in rare_classes[col_id]:
                        rare_cols_in_row.append(col_idx)

                if rare_cols_in_row:
                    # Include at least one rare column
                    num_rare_to_include = min(len(rare_cols_in_row), num_conditions)
                    selected_rare = np.random.choice(rare_cols_in_row, size=num_rare_to_include, replace=False)

                    # Fill remaining slots with random columns
                    remaining_slots = num_conditions - num_rare_to_include
                    if remaining_slots > 0:
                        available_cols = [idx for idx in range(len(col_spans)) if idx not in selected_rare]
                        if available_cols:
                            selected_regular = np.random.choice(
                                available_cols,
                                size=min(remaining_slots, len(available_cols)),
                                replace=False
                            )
                            selected_column_indices = np.concatenate([selected_rare, selected_regular])
                        else:
                            selected_column_indices = selected_rare
                    else:
                        selected_column_indices = selected_rare
                else:
                    # No rare columns, sample normally
                    selected_column_indices = np.random.choice(
                        len(col_spans),
                        size=num_conditions,
                        replace=False
                    )
            else:
                # Normal column selection
                selected_column_indices = np.random.choice(
                    len(col_spans),
                    size=num_conditions,
                    replace=False
                )

            batch_discrete_cols = []
            batch_category_ids = []

            # For selected columns, extract the REAL values from the sampled row
            for col_idx in selected_column_indices:
                col_id, st, ed = col_spans[col_idx]

                # Extract the real one-hot value from this row
                onehot_segment = data[row_id, st:ed]
                category_idx = np.argmax(onehot_segment)

                # Calculate global position
                global_id = self._discrete_column_cond_st[col_id] + category_idx

                # Set condition and mask
                cond[i, global_id] = 1.0
                mask[i, col_id] = 1.0

                batch_discrete_cols.append(col_id)
                batch_category_ids.append(category_idx)

            discrete_column_ids.append(batch_discrete_cols)
            category_ids_in_col.append(batch_category_ids)

        return cond, mask, discrete_column_ids, category_ids_in_col