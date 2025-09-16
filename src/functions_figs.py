import pandas as pd
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.formula.api as smf
import numpy as np

def print_lmem(df, pre_metric, post_metric):
    # Average correlation per (Patient, Radius)
    df_avg = (
        df[
            (df['Pre_Metric'] == pre_metric) &
            (df['Post_Metric'] == post_metric)
        ]
        .groupby(['Patient', 'Radius'])['Correlation']
        .mean()
        .reset_index()
    )

    # Fit LMEM on mean correlations
    model = smf.mixedlm("Correlation ~ Radius", df_avg, groups=df_avg["Patient"])
    result = model.fit()

    print(result.summary())
    return result, df_avg


def plot_subject_lines(df_avg, pre_metric, post_metric, result):
    # Extract stats
    beta = result.params['Radius']
    pval = result.pvalues['Radius']

    # Prepare colors
    patients = sorted(df_avg['Patient'].unique())
    cmap = cm.get_cmap('Blues')
    colors = [cmap(0.3 + 0.4 * i / (len(patients) - 1)) for i in range(len(patients))]

    # Plot
    plt.figure(figsize=(5.5, 3.4))
    for patient, color in zip(patients, colors):
        sub_df = df_avg[df_avg['Patient'] == patient]
        plt.plot(sub_df['Radius'], sub_df['Correlation'], marker='o', color=color, alpha=0.8)

    plt.xlabel("Radius",fontsize=12)
    plt.ylabel(r"$\langle \rho_s \rangle$",fontsize=12)
    plt.title(f"{pre_metric} (t<0) →  {post_metric} (t>0)",fontsize=14)
    plt.grid(True)

    # Annotate β and p
    plt.text(
        0.4, 0.075,
        rf"$\beta_{{\mathrm{{radius}}}} = {beta:.3f}$, " + (rf"$p = {pval:.3g}$"),
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    plt.tight_layout()
    #plt.savefig(path_out+'EEG_radius_%s-%s.png'%(pre_metric, post_metric),dpi=300)
    plt.show()
