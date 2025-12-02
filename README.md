# IMPUGAN

Simple GAN-based model for tabular imputation and conditional data
generation.

## 1. Training

You need a CSV with no missing values.

``` python
from core.impugan import IMPUGAN

model = IMPUGAN(...)
model.fit(train_df, discrete_columns)
```

## 2. Imputation

You need another CSV that contains missing values.

``` python
from run_impugan import run_impugan_train_and_impute

imputed_df = run_impugan_train_and_impute(
    train_df=train_df,
    masked_df=masked_df,
    discrete_columns=discrete_cols,
    model_kwargs=model_kwargs,
    output_csv_path="imputed.csv",
    scaler=scaler
)
```

## 3. Unconditional Generation

``` python
samples = model.sample(100)
```

## 4. Conditional Sampling (soft)

``` python
conds = [{"column": "gender", "value": "Female"}]
samples = model.sample(50, conditions=conds)
```

## 5. Hard Conditional Sampling (logits override)

``` python
conds = [{"column": "gender", "value": "Male"}]
rows = model.sample_hard_conditions(10, conditions=conds)
```

## Files Needed

- **available.csv** → Full training data. **This file must not contain any missing values.**  
If a row has any NaNs, move that row to the masked CSV instead.

-   **masked.csv** → file with missing values for imputation

## Running Full Pipeline

``` bash
python run_impugan.py
```
