import os
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from core.impugan import IMPUGAN


def make_conditions_from_row(row: pd.Series, discrete_cols: List[str], train_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extracts non-missing categorical values to use as sampling constraints."""
    conds = []
    for col, val in row.items():
        if pd.notna(val) and col in discrete_cols:
            if val in train_df[col].unique():
                conds.append({"column": col, "value": val})
    return conds


def run_impugan_train_and_impute(
        train_df: pd.DataFrame,
        masked_df: pd.DataFrame,
        discrete_columns: List[str],
        continuous_cols: List[str],
        model_kwargs: Optional[Dict[str, Any]] = None,
        output_csv_path: str = "imputed.csv",
        scaler: MinMaxScaler = None,
) -> pd.DataFrame:
    """
    Initializes, trains, and uses IMPUGAN to fill missing values.
    """
    # 1) Model Training Phase
    model_kwargs = model_kwargs or {}
    model = IMPUGAN(**model_kwargs)

    print(f"\n🚀 Starting training on {len(train_df)} samples...")
    model.fit(train_df, discrete_columns)

    # 2) Transition to Imputation
    print("\n" + "=" * 50)
    print("✅ TRAINING PHASE COMPLETE")
    print("=" * 50)

    total_missing = masked_df.isna().sum().sum()
    rows_to_impute = masked_df.isna().any(axis=1).sum()

    print(f"🔍 STARTING IMPUTATION")
    print(f"  - Filling {total_missing} total missing values")
    print(f"  - Processing {rows_to_impute} rows")
    print("-" * 50 + "\n")

    # 3) Imputation Phase
    out_df = masked_df.copy()
    for i, row in tqdm(out_df.iterrows(), total=len(out_df), desc="Imputing"):
        missing_cols = [col for col in row.index if pd.isna(row[col])]
        if not missing_cols:
            continue

        conds = make_conditions_from_row(row, discrete_columns, train_df)

        # Sample with hard constraints for discrete consistency
        syn = model.sample_hard_conditions(1, conditions=conds)

        if isinstance(syn, list):
            syn = pd.DataFrame(syn)

        if not syn.empty:
            sampled_row = syn.iloc[0]
            for col in missing_cols:
                if col in sampled_row.index:
                    out_df.at[i, col] = sampled_row[col]

    # 4) Post-processing and Export
    out_df = out_df.reset_index(drop=True)
    if continuous_cols and scaler:
        out_df[continuous_cols] = scaler.inverse_transform(out_df[continuous_cols])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)

    print(f"✅ Imputation complete. Results saved to: {output_csv_path}")
    return out_df


if __name__ == "__main__":
    # Example dataset configuration
    datasets = [{
        "dataset_name": "adult",
        "discrete_cols": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'native-country', 'income']
    }]

    for ds in datasets:
        name = ds['dataset_name']
        train = pd.read_csv(f"datasets/{name}/{name}_available.csv")
        masked = pd.read_csv(f"datasets/{name}/{name}_masked.csv")

        discrete_cols = ds['discrete_cols']
        continuous_cols = list(set(train.columns) - set(discrete_cols))

        scaler = MinMaxScaler()
        if continuous_cols:
            train[continuous_cols] = scaler.fit_transform(train[continuous_cols])
            masked[continuous_cols] = scaler.transform(masked[continuous_cols])

        model_params = {
            "embedding_dim": 128,
            "generator_dim": (256, 256),
            "discriminator_dim": (256, 256),
            "generator_lr": 2e-4,
            "batch_size": 128,
            "epochs": 2,
            "pac": 8
        }

        run_impugan_train_and_impute(
            train_df=train,
            masked_df=masked,
            discrete_columns=discrete_cols,
            continuous_cols=continuous_cols,
            model_kwargs=model_params,
            output_csv_path=f"imputed_files/{name}/imputed/results.csv",
            scaler=scaler
        )