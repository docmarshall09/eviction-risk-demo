# Yearly Logistic Regression Training Dataset Logic

## Training example definition
- `as_of_year` is the feature year `t` (stored as `year` in the feature table).
- `target_year` is `t + 1` (stored as `outcome_year`).
- Label `y` on the `as_of_year=t` row is `1` when that county is in the top quartile of `filing_rate` in `target_year=t+1`; otherwise `0`.

## Filter sequence (in order)
1. Load raw yearly CSV (`data/raw/county_proprietary_valid_2000_2018.csv`).
2. Clean raw rows (`clean_eviction_lab_yearly`):
   - Normalize `cofips -> county_fips` (zero-padded 5 digits).
   - Coerce `year`, `filings`, `filing_rate` to numeric.
   - Drop rows with invalid `county_fips`, `year`, or `filing_rate`.
   - Compute `sample_weight` from implied renter households; use `1.0` fallback for invalid rows and cap large weights at the 99th percentile.
3. Build yearly features (`build_eviction_lab_yearly_features`):
   - Sort by `county_fips`, `year`.
   - Create `lag_1`, `lag_3_mean_obs` (rolling 3 observed rows), `lag_5_mean_obs` (rolling 5 observed rows), `years_since_last_obs`.
   - Build shifted label `y` for next year and set `outcome_year = year + 1`.
   - Set `y` to null when the county has no observed row at `outcome_year`.
   - Drop rows missing any of: `lag_1`, `lag_3_mean_obs`, `lag_5_mean_obs`.
4. Final training-row filter (`build_yearly_training_dataset`, used by yearly training tasks):
   - Keep only rows with non-null:
     - `lag_1`
     - `lag_3_mean_obs`
     - `lag_5_mean_obs`
     - `years_since_last_obs`
     - `y`
     - `sample_weight`
     - `outcome_year`
   - Cast `y` to integer.

## Year range restrictions
- There is no hard-coded year filter in training-row construction.
- Effective years come from raw data availability and next-year label availability.
- For `--task train_eviction_lab_yearly`, model fitting then excludes the latest 2 feature years for holdout testing.
- For `--task train_eviction_lab_yearly_final`, all labeled rows are used (no holdout split).

## Missing value handling
- Missing critical raw fields (`county_fips`, `year`, `filing_rate`) are dropped in cleaning.
- Missing lag features are dropped in feature construction.
- Missing `y` (commonly no observed next-year county row) is dropped in final training-row filter.
- Missing/invalid exposure inputs do not drop rows directly; `sample_weight` falls back to `1.0`.

## De-duplication behavior
- No explicit de-duplication step is applied in the yearly training-dataset builder.

## Minimum-history constraints
- A county-year must have enough observed history to satisfy rolling windows:
  - at least 3 observed rows for `lag_3_mean_obs`
  - at least 5 observed rows for `lag_5_mean_obs`
- There is no extra minimum-history rule beyond these lag requirements.

