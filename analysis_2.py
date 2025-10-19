import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, ttest_ind, mannwhitneyu
import matplotlib

PROJECT_DIR = "project_data"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
matplotlib.use('Agg') 

# Second question ___________________________________________________________________________
misclassified = pd.read_csv(os.path.join(OUTPUT_DIR, "misclassified_points.csv"))
to_analyze = misclassified.select_dtypes(include = np.number).columns
results = []

for feat in to_analyze:
    truth = misclassified.groupby('truth')[feat].apply(list)
    pred = misclassified.groupby('prediction')[feat].apply(list)

    if misclassified["truth"].nunique() == 2: # just a safe net
        groups = [np.array(vals, dtype = float) for vals in truth]
        if len(groups) == 2:
            g1, g2 = groups
            print("g1:", g1[:5], "g2:", g2[:5])
            _, p1 = normaltest(g1)
            _, p2 = normaltest(g2)

            if p1 > 0.05 and p2 > 0.05:
                stat, pval = ttest_ind(g1, g2, equal_var=False)
                test_used = "t-test (truth)"
            else:
                stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                test_used = "Mann Whitney U (truth)"
            
            results.append({"feature": feat, "test": test_used, "p_value": pval})
    plt.figure()
    sns.kdeplot(data = misclassified, x = feat, hue = "truth", fill = True, common_norm = False)
    plt.title(f"{feat} distribution by truth (misclassified only)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{feat}_truth_distribution.svg"))
    plt.close()

    plt.figure()
    sns.kdeplot(data = misclassified, x = feat, hue = "prediction", fill = True, common_norm = False)
    plt.title(f"{feat} distribution by prediction (misclassified only)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{feat}_prediction_distribution.svg"))
    plt.close()

results_df = pd.DataFrame(results)
print(results_df)
