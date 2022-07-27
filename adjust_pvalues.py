import pandas as pd
from statsmodels.stats.multitest import multipletests

df_tests=pd.read_excel('results_statistical_analysis/run/collective_results/wilcoxon_results_all_weeks_before_adjustment.xlsx')
adj_res=multipletests(df_tests.pvalue,method='fdr_bh')
df_tests['adjusted_rejected']=adj_res[0]
df_tests['adjusted_pvalue']=adj_res[1]
df_tests.to_excel('results_statistical_analysis/run/collective_results/wilcoxon_results_all_weeks_adjusted.xlsx', index=False)
