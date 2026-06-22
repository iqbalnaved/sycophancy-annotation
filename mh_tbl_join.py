import pandas as pd

ex = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/clean_examples.csv')
mh = pd.read_csv('/mnt/d/Naved/analysis/helpmed/mental_health_detection.csv')

# Mapping from notebooks/streamlined_results.ipynb in the HELPMed repo
# Original has 'GPT 4o', changed to 'GPT-4o' to match paper formatting
treatments_map = {1: 'Llama 3 70B', 2: 'GPT-4o', 3: 'Control', 4: 'Command R+'}

merged = mh.merge(ex[['id', 'treatment_id', 'chat_history']], on='id', how='left')
merged['model'] = merged['treatment_id'].map(treatments_map)
merged = merged.drop(columns=['judge_raw', 'treatment_id'])

# Reorder so model and chat_history come after id
cols = ['id', 'model', 'chat_history', 'user_mental_health_detected', 'user_indicators',
        'user_severity', 'user_rationale', 'bot_response_quality', 'bot_rationale',
        'overall_concern_level']
merged = merged[cols]

merged.to_excel('/mnt/d/Naved/analysis/helpmed/mental_health_with_model.xlsx', index=False)