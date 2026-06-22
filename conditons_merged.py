import pandas as pd

ex = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/clean_examples.csv')
sc = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/scenarios.csv')
ps = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/clean_prescored.csv')
syc = pd.read_csv('/mnt/d/Naved/Outputs/helpmed/sycophancy.csv')

# Mapping from notebooks/streamlined_results.ipynb in the HELPMed repo
# Original has 'GPT 4o', changed to 'GPT-4o' to match paper formatting
treatments_map = {1: 'Llama 3 70B', 2: 'GPT-4o', 3: 'Control', 4: 'Command R+'}

merged = ex.merge(ps[['id', 'correct', 'differential_correct', 'red_flag_correct']], on='id')
merged = merged.merge(syc[['id', 'is_sycophantic', 'rationale']], on='id', how='left')
merged['model'] = merged['treatment_id'].map(treatments_map)

result = merged[['participant_id', 'model', 'chat_history', 'correct', 'differential_correct', 'red_flag_correct', 'is_sycophantic', 'rationale']]
result.columns = ['participant_id', 'model', 'chat_history', 'correct_disposition', 'diff_correct_condition', 'red_flag_correct_condition', 'is_sycophantic', 'rationale']

result.to_excel('/mnt/d/Naved/Data/HELPMed/participant_full_table.xlsx', index=False)