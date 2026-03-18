from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

root = Path('/Users/cbc/Documents/GitHub/fufo/notebook/DavideMomi/Revision/State_Dependent_Brain_Stimulation-main')
df = pd.read_csv(root / 'data/df_results/ieeg_metrics/df_correlations_5MOIs_ieeg.csv')
df = df[(df['Radius_post'] == 100) & (df['Metric_pre'] == 'ZCR') & (df['Metric_post'] == 'ZCR')].copy()
df['Session_ID'] = df['Participant'] + '_' + df['Session']
df['FisherZ'] = np.arctanh(np.clip(df['Correlation'], -0.999999, 0.999999))
print('rows', len(df), 'participants', df['Participant'].nunique(), 'sessions', df['Session_ID'].nunique())
model = smf.mixedlm(
    'FisherZ ~ Radius_pre',
    df,
    groups=df['Participant'],
    re_formula='~Radius_pre',
    vc_formula={'Session': '0 + C(Session_ID)'},
)
result = model.fit(method='lbfgs', reml=False, maxiter=200, disp=False)
print(result.summary())
