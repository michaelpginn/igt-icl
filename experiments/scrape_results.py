import os
import json
import pandas as pd

# Initialize an empty dictionary to hold the accuracy values
accuracy_dict = {}
tokens_dict = {}

# Define the base folder
lang = 'uspa1245'
strategy = 'word-recall'
base_folder = 'retrieval/' + strategy

# Iterate through each subfolder in the base folder
for subfolder in os.listdir(base_folder):
    # Check if the subfolder matches the expected format
    if subfolder.startswith(strategy + '-'):
        shots = int(subfolder.split('-')[-1])
        accuracy_dict[shots] = {}
        tokens_dict[shots] = {}
        
        # Define the path to the current subfolder
        current_folder = os.path.join(base_folder, subfolder)
        
        # Iterate through the expected files in the current subfolder
        for seed in range(3):
            file_name = f'{lang}.unseg.command-r-plus.{seed}.metrics.json'
            file_path = os.path.join(current_folder, file_name)
            
            # Check if the file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    accuracy = data.get('morpheme_level', {}).get('accuracy', None) * 100
                    accuracy_dict[shots][seed] = accuracy
                    tokens_dict[shots][seed] = data.get('avg_tokens')

# Convert the dictionary to a DataFrame for better visualization
accuracy_df = pd.DataFrame(accuracy_dict).transpose()
accuracy_df = accuracy_df.sort_index()
accuracy_df.to_csv(f'{lang}-{strategy}.csv')

tokens_df = pd.DataFrame(tokens_dict).transpose()
tokens_df = tokens_df.sort_index()
tokens_df.to_csv(f'{lang}-{strategy}.tokens.csv')
