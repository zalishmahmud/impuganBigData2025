from typing import List, Dict, Any, Optional
import os
from core.impugan import IMPUGAN
import pandas as pd
import sys



def make_conditions_from_row(row: pd.Series, discrete_cols, train_df) -> List[Dict[str, Any]]:
    """
    Given a row, build conditions = [{"column": col, "value": val}, ...]
    for all non-NaN columns.
    """
    conds = []
    unique_values_dict= {}
    # Convert to a list for cleaner printing, if needed
    for col, val in row.items():
        if pd.notna(val):
            unique_values_array = unique_values_dict.setdefault(col, train_df[col].unique().tolist())
            if col in discrete_cols:
                if val in unique_values_array:
                    conds.append({"column": col, "value": val})
    return conds


def run_impugan_train_and_impute(
    train_df: pd.DataFrame,
    masked_df: pd.DataFrame,
    discrete_columns: List[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
    output_csv_path: str = "imputed.csv",
    scaler = None,
) -> pd.DataFrame:
    """
    Train IMPUGAN on train_df, then impute masked_df row by row.
    Guarantees the output rows remain in the same order as masked_df.
    """
    # 1) Train model
    model_kwargs = model_kwargs or {}
    model = IMPUGAN(**model_kwargs)
    model.fit(train_df, discrete_columns)

    # 2) Copy masked_df for filling
    out_df = masked_df.copy()
    # 3) Iterate over rows with NaNs
    for i, row in out_df.iterrows():
        missing_cols = [col for col in row.index if pd.isna(row[col])]
        if not missing_cols:
            continue

        conds = make_conditions_from_row(row, discrete_columns, train_df)
        syn = model.sample_hard_conditions(1, conditions=conds)
        if isinstance(syn, list):
            syn = pd.DataFrame(syn)
        if syn.empty:
            continue

        sampled_row = syn.iloc[0]

        filled_cols = []
        for col in missing_cols:
            if col in sampled_row.index:
                out_df.at[i, col] = sampled_row[col]
                filled_cols.append(col)


    # 4) Reset index for strict alignment and save
    out_df = out_df.reset_index(drop=True)
    out_df[continuous_cols] = scaler.inverse_transform(out_df[continuous_cols])
    parent_directory = os.path.dirname(output_csv_path)
    os.makedirs(parent_directory, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    return out_df


import pandas as pd
import  json
from sklearn.preprocessing import MinMaxScaler

def json_to_dictionary(json_string):
    """
    Parses a JSON string into a Python list of dictionaries (uniform output).
    Handles both:
      - single dict {"column": ..., "value": ...}
      - list of dicts [{"column": ..., "value": ...}, ...]
    """
    try:
        parsed = json.loads(json_string)

        # Ensure always list of dicts
        if isinstance(parsed, dict):
            parsed = [parsed]
        elif not (isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed)):
            raise ValueError("Parsed JSON must be a dict or list of dicts.")

        print("✅ Successfully parsed JSON.")
        return parsed

    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None
    except (TypeError, ValueError) as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    datasets = [
        {
            "dataset_name": "adult",
            "discrete_cols":[
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country', 'income'
    ]
        }
    ]

    # ------------------------
    # PRE-VALIDATION
    # ------------------------
    print("🔍 Running dataset validation checks...")
    dirs = os.listdir('datasets')
    for dataset_hash in datasets:
        if not dataset_hash['dataset_name'] in dirs:
            raise Exception(f"Dataset {dataset_hash['dataset_name']} not found.")
    for dataset in datasets:
        train_path = f"datasets/{dataset['dataset_name']}/{dataset['dataset_name']}_available.csv"
        masked_path = f"datasets/{dataset['dataset_name']}/{dataset['dataset_name']}_masked.csv"

        try:
            train = pd.read_csv(train_path)
            masked = pd.read_csv(masked_path)
        except Exception as e:
            print(f"❌ Could not load CSVs for dataset={dataset['dataset_name']}: {e}")
            sys.exit(1)

        # Check discrete columns exist
        discrete_cols = dataset['discrete_cols']
        missing_in_train = [col for col in discrete_cols if col not in train.columns]
        missing_in_masked = [col for col in discrete_cols if col not in masked.columns]

        if missing_in_train or missing_in_masked:
            print(f"❌ Missing discrete columns in dataset={dataset['dataset_name']}")
            print(f"   Missing in train: {missing_in_train}")
            print(f"   Missing in masked: {missing_in_masked}")
            sys.exit(1)

        # Check no NaNs in training CSV
        if train.isna().any().any():
            nan_cols = train.columns[train.isna().any()].tolist()
            print(f"❌ NaN values found in train CSV for dataset={dataset['dataset_name']}")
            print(f"   Columns with NaN: {nan_cols}")
            sys.exit(1)

    print("✅ All validation checks passed. Proceeding to training...\n")

    # ------------------------
    # MAIN LOOP (your code stays here)
    # ------------------------
    for dataset in datasets:
        # Load CSVs
        train = pd.read_csv(
            f"datasets/{dataset['dataset_name']}/{dataset['dataset_name']}_available.csv"
        )
        masked = pd.read_csv(
            f"datasets/{dataset['dataset_name']}/{dataset['dataset_name']}_masked.csv"
        )

        # Define columns
        discrete_cols = dataset['discrete_cols']
        continuous_cols = list(set(train.columns.to_list()) - set(discrete_cols))

        # Initialize scaler
        scaler = MinMaxScaler()
        if continuous_cols:
            train[continuous_cols] = scaler.fit_transform(train[continuous_cols])
            masked[continuous_cols] = scaler.transform(masked[continuous_cols])

        # Model config
        model_kwargs = dict(
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            generator_decay=1e-6,
            discriminator_lr=2e-4,
            discriminator_decay=0,
            batch_size=128,
            epochs=300,
            pac=8
        )

        # Run training + imputation
        imputed = run_impugan_train_and_impute(
            train_df=train,
            masked_df=masked,
            discrete_columns=discrete_cols,
            model_kwargs=model_kwargs,
            output_csv_path=f"imputed_files/{dataset['dataset_name']}/imputed/imputed_with_impugan_run.csv",
            scaler=scaler,
        )

        print(f"✅ Imputation complete for dataset={dataset['dataset_name']}")