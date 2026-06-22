import pandas as pd

ex = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/clean_examples.csv')
sc = pd.read_csv('/mnt/d/Naved/Codes/HELPMed/data/main/scenarios.csv')

# Mapping from notebooks/streamlined_results.ipynb in the HELPMed repo
# Original has 'GPT 4o', changed to 'GPT-4o' to match paper formatting
treatments_map = {1: 'Llama 3 70B', 2: 'GPT-4o', 3: 'Control', 4: 'Command R+'}
ex['model'] = ex['treatment_id'].map(treatments_map)
merged = ex.merge(sc[['scenario_id', 'condition']], on='scenario_id')

with pd.ExcelWriter('/mnt/d/Naved/analysis/helpmed/scenario_conversations.xlsx', engine='openpyxl') as writer:
    for model in ['GPT-4o', 'Llama 3 70B', 'Command R+', 'Control']:
        sub = merged[merged['model'] == model]
        scenarios = sorted(sub['condition'].unique())

        rows = []
        max_cols = 0
        for scenario in scenarios:
            chats = sub[sub['condition'] == scenario]['chat_history'].tolist()
            max_cols = max(max_cols, len(chats))
            rows.append({'scenario': scenario, **{f'conv_{i+1}': c for i, c in enumerate(chats)}})

        df = pd.DataFrame(rows)
        cols = ['scenario'] + [f'conv_{i+1}' for i in range(max_cols)]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        df.to_excel(writer, sheet_name=model[:31], index=False)