import pandas as pd
import numpy as np
import scikit_posthocs as sp

'''
Performs the post hoc test on a specific measure
'''
def calc_post_hoc_results(df,measure):
    df['algorithm'] = np.nan
    df.iloc[::3, -1] = 'rf'
    df.iloc[1::3, -1] = 'swa'
    df.iloc[2::3, -1] = 'improved_swa'
    df = df.sort_values(by=['algorithm',measure])
    df = df.reset_index()
    df = df[[measure,'algorithm']]
    print(sp.posthoc_ttest(df,val_col=measure,pool_sd  = True,group_col='algorithm',p_adjust='hs'))

df_results = pd.read_csv(r".\report-csv.csv")
calc_post_hoc_results(df_results, 'Accuracy')