import pandas as pd 
import numpy as np

def match_controls(cohort_df, aki_col='aki_detected',
                   age_col='age_years', sex_col='sex',
                   ratio=2, seed=42):
    """
    For each AKI admission, sample `ratio` non-AKI admissions
    matched on age band and sex.
    
    Age bands:
        neonate  < 1 month
        infant   1 month – 1 year
        toddler  1 – 3 years
        child    3 – 12 years
        adolescent 12 – 18 years
    """
    def age_band(age):
        if age < 0.083:  return 'neonate'
        if age < 1:      return 'infant'
        if age < 3:      return 'toddler'
        if age < 12:     return 'child'
        return 'adolescent'

    cohort_df = cohort_df.copy()
    cohort_df['age_band'] = cohort_df[age_col].apply(age_band)
    cohort_df['strata']   = (
        cohort_df['age_band'] + '_' + cohort_df[sex_col].str.lower()
    )

    cases    = cohort_df[cohort_df[aki_col] == 1]
    patients_with_any_aki = set(
        cohort_df[cohort_df[aki_col]==1]['subject_id'].unique()
    )

    controls = cohort_df[~cohort_df['subject_id'].isin(patients_with_any_aki)]
    
    rng             = np.random.default_rng(seed)
    matched_control_ids = []
    unmatched       = 0

    for _, case in cases.iterrows():
        stratum = case['strata']
        pool    = controls[controls['strata'] == stratum]

        if len(pool) >= ratio:
            sampled = pool.sample(n=ratio, random_state=int(rng.integers(1e6)))
            matched_control_ids.extend(sampled['admission_id'].tolist())
        elif len(pool) > 0:
            # Not enough matches in stratum — take all available
            matched_control_ids.extend(pool['admission_id'].tolist())
            unmatched += 1
        else:
            # No match in stratum at all
            unmatched += 1

    if unmatched > 0:
        print(f'WARNING: {unmatched} cases could not be fully matched')

    matched_ids = (
        set(cases['admission_id'].tolist()) | 
        set(matched_control_ids)
    )
    
    matched_cohort = cohort_df[
        cohort_df['admission_id'].isin(matched_ids)
    ].copy()

    print(f'\nMatched cohort:')
    print(f'  AKI cases:    {matched_cohort[aki_col].sum()}')
    print(f'  Controls:     {(matched_cohort[aki_col]==0).sum()}')
    print(f'  Ratio:        '
          f'{(matched_cohort[aki_col]==0).sum() / matched_cohort[aki_col].sum():.2f}:1')
    
    # Verify matching balance
    print(f'\nAge band distribution:')
    print(matched_cohort.groupby([aki_col,'age_band']).size()
          .unstack(fill_value=0).to_string())
    
    return matched_cohort


# ── Usage ─────────────────────────────────────────────────
# Run on your cohort BEFORE building windows
# Then only build windows for matched admission_ids

cohort = pd.read_parquet('path/to/cohort.parquet') # load your cohort
cohort = cohort[~cohort['exclude_early_onset']].copy()
matched_cohort = match_controls(cohort, ratio=2)

# Filter windows to matched admissions only
windows = pd.read_parquet('path/to/windows.parquet') # update path as needed
matched_ids = set(matched_cohort['admission_id'])
windows_matched = windows[windows['admission_id'].isin(matched_ids)].copy()

print(f'\nWindows before matching: {len(windows)}')
print(f'Windows after matching:  {len(windows_matched)}')
print(f'New positive rate: {windows_matched["label"].mean()*100:.2f}%')

# UPDATE: Save matched cohort and windows to new files
windows_matched.to_parquet('path/to/windows_matched.parquet', index = False) 
matched_cohort.to_parquet('path/to/cohort_matched.parquet', index=False)
windows_matched.to_csv('path/to/windows_matched.csv', index = False)
matched_cohort.to_csv('path/to/cohort_matched.csv', index=False)