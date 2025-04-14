import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
import time
import os
start= time.time()

# set the script's location to be the current working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

############################## DATA IMPORTS ##############################
# exported from the "paper_submission_code.py" script
df_general_info_path = "../Data/Df_general_info.csv"
global_risk_path = "../Data/All_teams_Global_Returns.xlsx"
global_returns_path = "../Data/All_teams_Global_Risk.xlsx"


df_general_info = pd.read_csv(df_general_info_path, index_col=0)
global_risk_df = pd.read_excel(global_risk_path, index_col=0)
global_returns_df = pd.read_excel(global_returns_path, index_col=0)
##########################################################################

df_general_info["global_risk"] = global_risk_df["Global"]
df_general_info["global_returns"] = global_returns_df["Global"]
df_general_info = df_general_info.drop("Benchmark")
df_general_info.reset_index(inplace=True)


anova_test_df = df_general_info[["index", "Rank", "Global IR", "global_risk", "global_returns"]]

km = KMeans(n_clusters=3, random_state=2)
km.fit(anova_test_df[["global_risk"]])
# print(km.cluster_centers_)

threshold_1 = anova_test_df.global_risk.quantile(0.1)
threshold_2 = anova_test_df.global_risk.quantile(0.9)


# threshold_2 = float(anova_test_df.loc[anova_test_df["index"]=="Benchmark", "global_risk"])
# threshold_1 = 0.006

group_1 = anova_test_df.loc[anova_test_df.Rank<=threshold_1].index.tolist()
group_2 = anova_test_df.loc[((anova_test_df.Rank>threshold_1) & (anova_test_df.Rank<=threshold_2))].index.tolist()
group_3 = anova_test_df.loc[anova_test_df.Rank>threshold_2].index.tolist()

groups = [group_1, group_2, group_3]
# possible different submissions
possible_anova_crit_diff = pd.DataFrame(columns = ["Team A", "Team B", "Anova fstat", "Anova pvalue"])
ind = 0
for g1 in range(len(groups)):
    for g2 in range(len(groups)):
        
        group1 = groups[g1]
        group2 = groups[g2]
        
        possible_anova_crit_diff.loc[ind, "Team A"] = g1
        possible_anova_crit_diff.loc[ind, "Team B"] = g2
        possible_anova_crit_diff.loc[ind, "Risk_A_mean"] = anova_test_df.loc[group1, "Rank"].mean()
        possible_anova_crit_diff.loc[ind, "Risk_B_mean B"] = anova_test_df.loc[group2, "Rank"].mean()
        possible_anova_crit_diff.loc[ind, "Anova fstat"] = f_oneway(anova_test_df.loc[group1, "Rank"].values, anova_test_df.loc[group2, "Rank"].values)[0]
        possible_anova_crit_diff.loc[ind, "Anova pvalue"] = f_oneway(anova_test_df.loc[group1, "Rank"].values, anova_test_df.loc[group2, "Rank"].values)[1]
        ind += 1

# check whether there is any significant difference between two different groups
temp = possible_anova_crit_diff[possible_anova_crit_diff["Anova pvalue"]<0.05]

print(temp)

end = time.time()
print(end - start)

# 1 second