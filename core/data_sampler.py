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

        self._rid_by_cat_cols = []
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

        print(f"DataSampler initialized:")
        print(f"  - Total discrete columns: {self._n_discrete_columns}")
        print(f"  - Total categories: {self._n_categories}")
        print(f"  - Data length: {self._data_length}")

        for i in range(self._n_discrete_columns):
            n_cats = self._discrete_column_n_category[i]
            print(f"  - Column {i}: {n_cats} categories")
            category_labels = None
            if hasattr(self, "transformer") and self.transformer is not None:
                try:
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
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.random.rand()
        cumsum = np.cumsum(probs[:self._discrete_column_n_category[discrete_column_id]])
        return np.searchsorted(cumsum, r)

    def sample_condvec_multi(self, batch, num_conditions=None):
        if self._n_discrete_columns == 0:
            return None

        if num_conditions is None:
            max_possible = int(self._n_discrete_columns * (1 / 3))
            num_conditions = np.random.choice([1, 2, max_possible], p=[0.3, 0.5, 0.2])

        num_conditions = min(num_conditions, self._n_discrete_columns)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        discrete_column_ids = []
        category_ids_in_col = []

        for i in range(batch):
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

    def sample_data_multi_cond_improved(self, data, n, col_lists, opt_lists):
        if col_lists is None or all(len(cols) == 0 for cols in col_lists):
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        for cols, opts in zip(col_lists, opt_lists):
            if len(cols) == 0:
                idx.append(np.random.randint(len(data)))
            else:
                valid_rows = None
                condition_sets = []
                for c, o in zip(cols, opts):
                    rows_for_condition = set(self._rid_by_cat_cols[c][o])
                    condition_sets.append(rows_for_condition)
                    if valid_rows is None:
                        valid_rows = rows_for_condition
                    else:
                        valid_rows = valid_rows.intersection(rows_for_condition)

                if valid_rows and len(valid_rows) > 0:
                    idx.append(np.random.choice(list(valid_rows)))
                else:
                    best_intersection = set()
                    for i in range(len(condition_sets)):
                        for j in range(i + 1, len(condition_sets)):
                            pair_intersection = condition_sets[i].intersection(condition_sets[j])
                            if len(pair_intersection) > len(best_intersection):
                                best_intersection = pair_intersection

                    if len(best_intersection) > 0:
                        idx.append(np.random.choice(list(best_intersection)))
                    else:
                        largest_set = max(condition_sets, key=len)
                        if len(largest_set) > 0:
                            idx.append(np.random.choice(list(largest_set)))
                        else:
                            idx.append(np.random.randint(len(data)))
        return data[idx]

    def generate_multi_cond_from_conditions(self, conditions, batch, transformer):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        for condition in conditions:
            try:
                info = transformer.convert_column_name_value_to_id(condition['column'], condition['value'])
                col_id = info['discrete_column_id']
                value_id = info['value_id']
                global_id = self._discrete_column_cond_st[col_id] + value_id
                vec[:, global_id] = 1
                mask[:, col_id] = 1
            except Exception:
                continue
        return vec, mask

    def check_multi_condition_feasibility(self, conditions, transformer):
        if not conditions:
            return self._data_length, 100.0
        valid_rows = None
        for condition in conditions:
            try:
                info = transformer.convert_column_name_value_to_id(condition['column'], condition['value'])
                col_id = info['discrete_column_id']
                value_id = info['value_id']
                rows_for_condition = set(self._rid_by_cat_cols[col_id][value_id])
                valid_rows = rows_for_condition if valid_rows is None else valid_rows.intersection(rows_for_condition)
            except Exception:
                return 0, 0.0
        if valid_rows is None:
            return 0, 0.0
        count = len(valid_rows)
        return count, (count / self._data_length * 100)

    def dim_cond_vec(self):
        return self._n_categories

    def build_condvec_from_user_conditions(self, transformer, conditions, batch):
        if not conditions:
            return None, None, [], []
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        cols, opts = [], []
        for condition in conditions:
            info = transformer.convert_column_name_value_to_id(condition['column'], condition['value'])
            col_id = info['discrete_column_id']
            val_id_in_col = info['value_id']
            global_cat_id = self._discrete_column_cond_st[col_id] + val_id_in_col
            cond[:, global_cat_id] = 1.0
            mask[:, col_id] = 1.0
            cols.append(col_id)
            opts.append(val_id_in_col)
        return cond, mask, cols, opts