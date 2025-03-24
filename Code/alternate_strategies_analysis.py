import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# set current working directory
os.chdir("C:/Users/TasosK/Desktop/Tasos/A-Research/M6 Discussion Paper/Submission_Workspace")

############################## DATA IMPORTS ##############################
# exported from the "alternate_strategies_file.py" script
overview_df_path = "./Data/Alternate_strategies_final_IR.csv"
# exported from the "paper_submission_code.py" script
df_general_info_path = "./Data/Df_general_info.csv"

##########################################################################

overview_df = pd.read_csv(overview_df_path, index_col=0)
df_general_info = pd.read_csv(df_general_info_path, index_col=0)

overview_df = overview_df.loc[df_general_info.index]
overview_df["# Submissions"] = df_general_info["Unique_Submissions"]


Delta_df = pd.DataFrame(index=list(range(1,13)), columns=range(1,13))
Counts_df = pd.DataFrame(index=list(range(1,13)), columns=range(1,13))
Rank_df = pd.DataFrame(index=list(range(1,13)), columns=range(1,13))

teams_w_n_subs = overview_df[overview_df["# Submissions"]==12]
teams_w_n_subs.iloc[:,:12] = teams_w_n_subs.iloc[:,:12].sub(teams_w_n_subs['Global'], axis=0)


for sub_cnt in range(1,13):

    teams_w_n_subs = overview_df[overview_df["# Submissions"]==sub_cnt]
    teams_w_n_subs.iloc[:,:12] = teams_w_n_subs.iloc[:,:12].sub(teams_w_n_subs['Global'], axis=0)
    count_df_tmp = teams_w_n_subs.copy()
    count_df_tmp[count_df_tmp > 0] = 1
    count_df_tmp[count_df_tmp < 0] = 0
    rank_df_tmp = teams_w_n_subs.copy()
    rank_df_tmp.iloc[:,:12] = rank_df_tmp.iloc[:,:12].rank(axis=1, ascending=False)
    
    Delta_df.loc[sub_cnt] = teams_w_n_subs.mean()[:12].values
    Counts_df.loc[sub_cnt] = count_df_tmp.mean()[:12].values
    Rank_df.loc[sub_cnt] = rank_df_tmp.mean()[:12].values


idy = np.r_[0:11, 12]
idx = np.r_[0:4, 5, 9, 11]
top_x_teams = overview_df.iloc[idx, idy]
top_x_teams.index = [1,2,3,4,6,10,12]
top_x_teams.columns = list(range(1,13))
top_x_teams = top_x_teams.T
top_x_teams = top_x_teams-top_x_teams.iloc[-1,:]

 
plt.cla()
plt.clf()
a4_dims = (9, 9)
plt.figure(figsize=a4_dims)
palette = sns.color_palette("Blues",n_colors=7)
palette.reverse()
g = sns.relplot(top_x_teams, kind='line',
              facet_kws={'despine':False}, palette = palette,
              dashes=False)
g._legend.set_title("Rank")
g._legend.set_bbox_to_anchor([1.02,0.5])
g._legend.borderaxespad = 1
plt.xlabel('Number of submissions made', fontsize=12)
plt.ylabel('Estimated IR', fontsize=12)
plt.xticks(list(range(1,13)), fontsize=8)
plt.yticks(fontsize=8)

plt.gca().set_xticklabels([str(x) for x in list(range(1,13))])
plt.grid()

plt.savefig("./Results/SV_top_x_alternate.eps", format='eps')
plt.savefig("./Results/SV_top_x_alternate.png")


       
plot_data = Delta_df.iloc[[2,5,8,11], :11]
plot_data = plot_data.T

plt.cla()
plt.clf()
a4_dims = (12, 6)
plt.figure(figsize=a4_dims)
palette = sns.color_palette("Blues",n_colors=4)
g=sns.relplot(plot_data, kind='line',
              facet_kws={'despine':False}, palette = palette,
              dashes=False, height=5, aspect= 1.33)
g._legend.set_title("# Submissions")
g._legend.set_bbox_to_anchor([1.15,0.5])
g.legend.get_title().set_fontsize(14)
plt.setp(g._legend.get_texts(), fontsize=12)
plt.xlabel('Number of submissions made', fontsize=14)
plt.ylabel('Estimated IR difference', fontsize=14)
plt.xticks(list(range(1,12)), fontsize=12)
plt.yticks(fontsize=12)
plt.gca().set_xticklabels([str(x) for x in list(range(1,12))])
plt.grid()

plt.savefig("./Results/SV_avg_delta_alternate_strategies.eps", bbox_inches='tight', format='eps')
plt.savefig("./Results/SV_avg_delta_alternate_strategies.png", bbox_inches='tight')

# regression to get the r and p values for the group of 12 submissions
# They are shown in the manuscript as a footnote (number 3) reffering to the clear trend that the plot shows
print("********************************************")
print("**** Group of 12 Submissions Regression ****")
print("********************************************")
X2 = sm.add_constant(plot_data.index.astype(float))
est = sm.OLS(plot_data[12].astype(float), X2)
est2 = est.fit()
print(est2.summary())


# regression to get the r and p values for the regression of the winner's alternate strategies
# They are shown in the manuscript as a footnote (number 4)
print("********************************************")
print("** Winner Alternate Strategies Regression **")
print("********************************************")
X2 = sm.add_constant(top_x_teams.index.astype(float))
est = sm.OLS(top_x_teams.iloc[:,0].astype(float), X2)
est2 = est.fit()
print(est2.summary())










