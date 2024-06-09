import os
import re
import pandas as pd
from pathlib import Path
import datasets
import fire
from run_experiment import _create_gloss_list
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 20         # Default font size for text
mpl.rcParams['axes.titlesize'] = 24     # Font size for the title
mpl.rcParams['axes.labelsize'] = 20    # Font size for x and y labels
mpl.rcParams['xtick.labelsize'] = 20    # Font size for x tick labels
mpl.rcParams['ytick.labelsize'] = 16    # Font size for y tick labels
mpl.rcParams['legend.fontsize'] = 16    # Font size for legend
mpl.rcParams['legend.title_fontsize'] = 16  # Font size for legend title

def count_glosses_in_preds(preds_path, glosslist):
    df = pd.read_csv(preds_path, sep='\t')
    

    def count_glosses(gloss_string, glosslist):
        if not isinstance(gloss_string, str):
            return 0
        
        gloss_count = 0
        for gloss in re.split(r'[ -]', gloss_string):
            if gloss.isupper() and gloss in glosslist:
                gloss_count += 1
        return gloss_count

    def count_total_glosses(gloss_string):
        if not isinstance(gloss_string, str):
            return 0
        gloss_count = 0
        for gloss in re.split(r'[ -]', gloss_string):
            if gloss.isupper():
                gloss_count += 1
        return gloss_count

    df['gloss_count'] = df['predicted_glosses'].apply(lambda x: count_glosses(x, glosslist))
    df['total_glosses'] = df['predicted_glosses'].apply(lambda x: count_total_glosses(x))

    overall_gloss_count = df['gloss_count'].sum()
    overall_total_glosses = df['total_glosses'].sum()

    return (overall_gloss_count / overall_total_glosses * 100) if overall_total_glosses > 0 else 0

def measure_adherence_for_folder(folder_path: str, used_glosslist):
    files = list(Path(folder_path).rglob('**/*.preds.tsv'))
    adherence_data = []

    glosslm_corpus = datasets.load_dataset("lecslab/glosslm-corpus-split")
    glottocodes_ID = set(glosslm_corpus['eval_ID']['glottocode'])
    glottocodes_OOD = set(glosslm_corpus['eval_OOD']['glottocode'])

    for file in tqdm(files, desc=f"Processing files in {folder_path}"):
        glottocode = file.stem.split('.')[0]
        
        if glottocode in glottocodes_ID:
            id_or_ood = "ID"
        elif glottocode in glottocodes_OOD:
            id_or_ood = "OOD"
        else:
            continue

        glosslm_corpus_filtered = glosslm_corpus.filter(lambda row: row["is_segmented"] == "no")
        train_dataset = glosslm_corpus_filtered[f"train_{id_or_ood}"].filter(lambda row: row['glottocode'] == glottocode)

        language = {
            'lezg1247': 'Lezgi',
            'natu1246': 'Natugu',
            'uspa1245': 'Uspanteko',
            'gitx1241': 'Gitksan'
        }[glottocode]
        
        gloss_list = _create_gloss_list("split_morphemes", train_dataset)

        adherence = count_glosses_in_preds(file, gloss_list)
        adherence_data.append({
            'language': language,
            'file': str(file),
            'adherence': adherence,
            'used_glosslist': "+Glosslist" if used_glosslist else "Base"
        })

    return adherence_data

def main():
    glosslist_folder = './experiments/shots/glosslist'
    no_glosslist_folder = './experiments/shots/no-glosslist'

    glosslist_data = measure_adherence_for_folder(glosslist_folder, True)
    no_glosslist_data = measure_adherence_for_folder(no_glosslist_folder, False)

    all_data = glosslist_data + no_glosslist_data

    df = pd.DataFrame(all_data)
    # df.to_csv('adherence_values.csv', index=False)
    # print("Adherence values saved to adherence_values.csv")

    # Plotting
    plt.figure(figsize=(7, 6))
    sns.boxplot(x='language', y='adherence', hue='used_glosslist', data=df, palette=["#254653", "#299D8F"], hue_order=['Base', '+Glosslist'], order=['Gitksan', 'Lezgi', 'Natugu', 'Uspanteko'])
    plt.xlabel('')
    plt.ylabel('Adherence Percentage')

    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_edgecolor('#D1D1D1')
        spine.set_linewidth(1.5)

    # Remove all tick marks while keeping tick labels
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=1.5, color="#D1D1D1")
    plt.savefig('adherence_boxplot.pdf')
    plt.show()
    print("Boxplot saved as adherence_boxplot.pdf")


if __name__ == "__main__":
    fire.Fire(main)