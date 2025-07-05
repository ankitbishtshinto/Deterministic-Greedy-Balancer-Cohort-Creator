import pandas as pd
import numpy as np
import math


class DeterministicGreedyBalancer:
    def __init__(
        self,
        processing_columns: list,
        cohort_id_column: str,
        cohort_splits: list,
        error_margin: float = 0.05,
        weights=None,
    ):
        self._processing_columns = processing_columns
        self._cohort_id_column = cohort_id_column
        self._cohort_splits = cohort_splits
        self._error_margin = error_margin
        self._weights = weights

        total_split = sum(c['split_percentage'] for c in cohort_splits)
        if not np.isclose(total_split, 1.0, atol=1e-6):
            raise ValueError(
                f"Sum of all 'split_percentage' must be 1.0 (got {total_split:.4f})."
            )

        for c in cohort_splits:
            print(f"  - {c['cohort_name']}: {c['split_percentage']:.2f}")

    def process_data(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        df[self._processing_columns] = (
            df[self._processing_columns]
            .apply(pd.to_numeric, errors='coerce')
            .astype(float)
        )
        df.dropna(subset=self._processing_columns, inplace=True)

        if df.empty:
            return {
                c['cohort_name']: pd.DataFrame()
                for c in self._cohort_splits
            }

        df = self.normalize_parameters(df, self._processing_columns)
        norm_cols = [col + "_normalized" for col in self._processing_columns]
        df = self.compute_composite_score(df, norm_cols, self._weights)
        df = self.sort_users(df)
        return self.assign_users_to_cohorts(df, norm_cols)

    def normalize_parameters(
        self,
        df: pd.DataFrame,
        params: list,
    ) -> pd.DataFrame:
        for col in params:
            if df[col].nunique() <= 1:
                df[col + "_normalized"] = 0.0
            else:
                mn, mx = df[col].min(), df[col].max()
                if mx > mn:
                    df[col + "_normalized"] = (df[col] - mn) / (mx - mn)
                else:
                    df[col + "_normalized"] = 0.0
        return df

    def compute_composite_score(
        self,
        df: pd.DataFrame,
        norm_cols: list,
        weights=None,
    ) -> pd.DataFrame:
        if weights is None:
            weights = [1.0] * len(norm_cols)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        df["composite_score"] = df[norm_cols].dot(weights)
        return df

    def sort_users(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._cohort_id_column in df.columns:
            return (
                df.sort_values(
                    by=["composite_score", self._cohort_id_column],
                    ascending=[False, True],
                )
                .reset_index(drop=True)
            )
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    def assign_users_to_cohorts(
        self,
        df: pd.DataFrame,
        norm_cols: list,
    ) -> dict:
        N = len(df)
        k = len(self._cohort_splits)
        float_targets = [c['split_percentage'] * N for c in self._cohort_splits]
        floor_targets = [int(math.floor(t)) for t in float_targets]
        leftover = N - sum(floor_targets)

        # Distribute leftover counts by largest fractional parts
        frac_list = sorted(
            [
                (float_targets[i] - floor_targets[i], i)
                for i in range(k)
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        for i in range(leftover):
            floor_targets[frac_list[i][1]] += 1

        # Compute min/max allowed per cohort
        if np.isclose(self._error_margin, 0.0):
            min_allowed = floor_targets[:]
            max_allowed = floor_targets[:]
        else:
            min_allowed = [
                int(math.floor(float_targets[i] * (1 - self._error_margin)))
                for i in range(k)
            ]
            max_allowed = [
                int(math.ceil(float_targets[i] * (1 + self._error_margin)))
                for i in range(k)
            ]

        assigned_records = {
            c['cohort_name']: [] for c in self._cohort_splits
        }
        assigned_counts = [0] * k
        sums_of_cols = [[0.0] * len(norm_cols) for _ in range(k)]
        total_col_sums = [df[col].sum() for col in norm_cols]

        def distribution_cost(matrix):
            cost = 0.0
            for idx, cdict in enumerate(self._cohort_splits):
                desired = cdict['split_percentage']
                for j in range(len(norm_cols)):
                    actual = (
                        matrix[idx][j] / total_col_sums[j]
                        if total_col_sums[j]
                        else 0.0
                    )
                    cost += (actual - desired) ** 2
            return cost

        for _, row in df.iterrows():
            best_idx, best_cost = None, float('inf')
            for i in range(k):
                if assigned_counts[i] < max_allowed[i]:
                    assigned = sum(assigned_counts)
                    left = N - assigned - 1
                    need = sum(
                        max(0, min_allowed[j] - assigned_counts[j])
                        for j in range(k) if j != i
                    )
                    if need > left:
                        continue

                    # Tentatively add this row’s normalized values
                    original = sums_of_cols[i][:]
                    for j, col in enumerate(norm_cols):
                        sums_of_cols[i][j] += row[col]
                    cost = distribution_cost(sums_of_cols)
                    sums_of_cols[i] = original

                    if cost < best_cost:
                        best_cost, best_idx = cost, i

            if best_idx is None:
                choices = [
                    i for i in range(k)
                    if assigned_counts[i] < max_allowed[i]
                ]
                best_idx = (
                    max(
                        choices,
                        key=lambda x: self._cohort_splits[x]['split_percentage']
                    )
                    if choices else 0
                )

            cname = self._cohort_splits[best_idx]['cohort_name']
            assigned_records[cname].append(row.to_dict())
            assigned_counts[best_idx] += 1
            for j, col in enumerate(norm_cols):
                sums_of_cols[best_idx][j] += row[col]

        return {
            name: (
                pd.DataFrame(records)
                .assign(type=name)
                if records
                else pd.DataFrame()
            )
            for name, records in assigned_records.items()
        }


def build_cohort_comb_ind(
    df: pd.DataFrame,
    cohorts: list,
    metric_columns: list,
    error_margin: float = 0.01,
) -> pd.DataFrame:
    total = sum(c['split_percentage'] for c in cohorts)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Sum of split percentages must be 1.0 (got {total:.4f})."
        )

    output = []
    for et in df['experiment_type'].unique():
        tmp = df[df['experiment_type'] == et].copy()
        cohort_id_col = (
            'agency-hotel'
            if et.startswith('Comb GRP')
            else 'hotel_code'
        )
        balancer = DeterministicGreedyBalancer(
            processing_columns=metric_columns,
            cohort_id_column=cohort_id_col,
            cohort_splits=cohorts,
            error_margin=error_margin,
        )
        assigned = balancer.process_data(tmp)
        for c in cohorts:
            name = c['cohort_name']
            df_grp = assigned[name]
            if not df_grp.empty:
                df_grp = df_grp.rename(columns={'type': 'group'})
            output.append(df_grp)

    return pd.concat(output, axis=0)
