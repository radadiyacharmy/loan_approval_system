#%%

import pandas as pd 
import numpy as np
import os

#%%

parent_dir = os.path.dirname(os.getcwd())

results_folder_path = os.path.join(parent_dir, 'results')

base_df_path = os.path.join(parent_dir, 'base_data', 'on_us_performance_home_loan_applications.csv')

base_df = pd.read_csv(base_df_path)

#%%

# Define parameter lists specified by your requirements
mob_list = [3, 6, 9, 12, 15, 18, 24, 30, 36, 48]
dpd_list = [30, 60, 90]
max_dpd_mob_list = [3, 6, 9, 12, 18, 24, 30, 36, 48]

# 2. Initialize application-level summary dataframe
summary_df = pd.DataFrame({'app_id': base_df['app_id'].unique()})

# 3. Calculate max_mob for each application_id
grouped = base_df.groupby('app_id')
summary_df['max_mob'] = grouped['mob'].max().values

# 4. Calculate max_dpd in first N MOBs
for m in max_dpd_mob_list:
    sub = base_df[base_df['mob'] <= m]
    max_dpd_n = sub.groupby('app_id')['dpd'].max()
    summary_df[f'max_dpd_in_first_{m}_mob'] = summary_df['app_id'].map(max_dpd_n).fillna(0).astype(int)

# 5. Calculate flags: ever_x_dpd_y_mob
# (1 if application became >= x DPD in first y MOBs, else 0)
for x in dpd_list:
    for y in mob_list:
        sub = base_df[base_df['mob'] <= y]
        ever_flag = sub.groupby('app_id')['dpd'].apply(lambda s: (s >= x).any())
        summary_df[f'ever_{x}_dpd_{y}_mob'] = summary_df['app_id'].map(ever_flag).astype(int)

# 6. Save to CSV
output_filename = "bad_def_summary.csv"
summary_df.to_csv(os.path.join(results_folder_path, output_filename), index=False)

#%%


mob_list = [3, 6, 9, 12, 15, 18, 24, 30, 36, 48]

# Bad definitions: all ever_flags EXCEPT 90 DPD (i.e. 30 and 60 DPD across all MOBs)
bad_dpd_list = [30, 60]
# NPA definitions: all 90 DPD flags across all MOBs
npa_dpd_list = [90]

rows = []

# 2. Iterate through all pairwise combinations between bad_def and npa_def
for y_bad in mob_list:
    for x_bad in bad_dpd_list:
        bad_col = f"ever_{x_bad}_dpd_{y_bad}_mob"
        
        # Eligible population for Bad Definition
        bad_eligible_mask = ((summary_df['max_mob'] > y_bad) & (summary_df[bad_col] != 1)) | (summary_df[bad_col] == 1)
        total_count_bad = int(bad_eligible_mask.sum())
        bad_count = int(summary_df.loc[bad_eligible_mask, bad_col].sum())
        
        for y_npa in mob_list:
            for x_npa in npa_dpd_list:
                npa_col = f"ever_{x_npa}_dpd_{y_npa}_mob"
                
                # Eligible population for NPA Definition
                npa_eligible_mask = ((summary_df['max_mob'] > y_npa) & (summary_df[npa_col] != 1)) | (summary_df[npa_col] == 1)
                total_count_npa = int(npa_eligible_mask.sum())
                npa_count = int(summary_df.loc[npa_eligible_mask, npa_col].sum())
                
                # Intersection of bad_def == 1 and npa_def == 1
                bad_to_npa = int(summary_df[(summary_df[bad_col] == 1) & (summary_df[npa_col] == 1)].shape[0])
                bad_to_npa_rate = round(bad_to_npa / bad_count, 6) if bad_count > 0 else 0.0
                
                rows.append({
                    'bad_mob_window': y_bad,
                    'bad_def': bad_col,
                    'npa_mob_window': y_npa,
                    'npa_def': npa_col,
                    'total_count_bad_def': total_count_bad,
                    'bad_count': bad_count,
                    'bad_rate': round(bad_count / total_count_bad, 6) if total_count_bad > 0 else 0.0,
                    'total_count_npa_def': total_count_npa,
                    'npa_count': npa_count,
                    'npa_rate': round(npa_count / total_count_npa, 6) if total_count_npa > 0 else 0.0,
                    'bad_to_npa': bad_to_npa,
                    'bad_to_npa_rate': bad_to_npa_rate
                })

result_df = pd.DataFrame(rows)

#%%

result_df = pd.DataFrame(rows)

result_df['bad_mob'] = result_df['bad_def'].str.split('_').str[3]
result_df['bad_mob'] = result_df['bad_mob'].astype('int')

result_df['npa_mob'] = result_df['npa_def'].str.split('_').str[3]
result_df['npa_mob'] = result_df['npa_mob'].astype('int')

result_df = result_df[result_df['npa_mob'].isin([30, 36, 48])]

result_df = result_df[result_df['bad_mob'] < 30]

#%%

result_df['capture/recall'] = (result_df['bad_to_npa'] / result_df['npa_count']) * 100

result_df['conversion/precision'] = (result_df['bad_to_npa'] / result_df['bad_count']) * 100

result_df['f1_score'] = 2 * (result_df['capture/recall'] * result_df['conversion/precision']) / (result_df['capture/recall'] + result_df['conversion/precision'])

#%%

result_df.to_csv(os.path.join(results_folder_path, "bad_npa_definition_summary.csv"), index=False) 

#%%     