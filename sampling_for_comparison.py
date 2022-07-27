import pandas as pd
import os

def main():
    data_folder = os.path.join('data', 'symptoms_expressions')
    input_fpath = os.path.join(data_folder, 'filtered_tweets_medcat_lexicon.tsv')

    filtered_df = pd.read_csv(input_fpath, sep='\t')
    
    # Shuffle rows
    filtered_df = filtered_df.sample(frac=1, random_state=1)
    # Make sure that users represented in the sample contribute to one tweet at most
    filtered_df.drop_duplicates(subset=['user.id'], inplace=True)
    # Extract 500 tweets
    filtered_df = filtered_df.iloc[:500,:]

    v_pos = list(filtered_df.columns).index('vaccinated')
    symp_list = filtered_df.columns[v_pos+1:]
    # Create empty DataFrame
    df_compare = pd.DataFrame(columns=['text','med_symp','lex_symp'])
    # Fill DataFrame with symptom tags determined by MedCAT and the lexicon respectively
    for i,(idx, row) in enumerate(filtered_df.iterrows()):
        med_list = []
        lex_list = []
        for symp in symp_list:
            if row[symp] == 1:
                if symp[:3] == 'Lex':
                    lex_list.append(symp[4:])
                else:
                    med_list.append(symp)
        if len(med_list) or len(lex_list):
            df_compare.loc[i,'text'] = row['text']
            df_compare.loc[i,'med_symp'] = med_list
            df_compare.loc[i,'lex_symp'] = lex_list
    
    # Keep 100 tweets
    df_compare = df_compare.iloc[:100, :]

    # Save results
    output_folder = 'data'
    output1_fpath = os.path.join(output_folder, 'df_compare.tsv')
    output2_fpath = os.path.join(output_folder, 'df_compare_blind.tsv')
    # File with all columns
    df_compare.to_csv(output1_fpath, index=False, sep='\t')
    # File with text only
    df_compare.text.to_csv(output2_fpath, index=False, sep='\t')



if __name__ == '__main__':
    main()
