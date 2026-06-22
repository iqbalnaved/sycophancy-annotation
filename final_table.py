import pandas as pd

INPUT = '/mnt/d/Naved/Outputs/helpmed/sycophancy_run0_gpt-5.2-2025-12-11_02262026_1210.csv'
OUTPUT = '/mnt/d/Naved/Outputs/helpmed/participant_interactions_sycophancy_run0_gpt-5.2-2025-12-11_02262026_1210.xlsx'


# clean_examples.csv columns: [generated_datetime    scenario_id    id    treatment_id    controlText    historical_responses_model    chat_history    likely_cause    next_step    next_step_conf    participant_id] 
# scenarios.csv columns: [scenario_id    urgency    red_flags    full_differential    condition    conditions_short    conditions_order    urgency_text    medqa_queries]
ex = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/clean_examples.csv')
sc = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/scenarios.csv')
syc = pd.read_csv(INPUT)

# Mapping from notebooks/streamlined_results.ipynb in the HELPMed repo
# Original has 'GPT 4o', changed to 'GPT-4o' to match paper formatting
treatments_map = {1: 'Llama 3 70B', 2: 'GPT-4o', 3: 'Control', 4: 'Command R+'}
label_map = {
    'Ambulance (I am in immediate life-threatening danger; I need treatment administered en-route to the hospital)': 'Ambulance',
    'A&E (I need emergency hospital treatment)': 'A&E',
    'Urgent Primary Care (I should be seen today, by a GP, urgent care centre, or similar)': 'Urgent Primary Care',
    'Routine GP (I should be seen at some point, but it can wait)': 'Routine GP',
    'Self-care (I can handle this at home or with over-the-counter medication)': 'Self-care'
}

merged = ex.merge(sc[['scenario_id', 'urgency_text']], on='scenario_id') #  joins ground truth urgency label onto each example via scenario_id.
merged = merged.merge(syc[['id', 'is_sycophantic', 'rationale']], on='id', how='left') # id = 
merged['model'] = merged['treatment_id'].map(treatments_map)        # treatment_id = model
merged['user_disposition'] = merged['next_step'].map(label_map)     # next_step = user_disposition ie. prediction
merged['gold_disposition'] = merged['urgency_text'].map(label_map) # urgency_text = gold_disposition ie. ground truth

result = merged[['participant_id', 'model', 'chat_history', 'user_disposition', 'gold_disposition', 'is_sycophantic', 'rationale']]
result.to_excel(OUTPUT, index=False)