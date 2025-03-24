######################################################################################################################
##### Contains the code for exporting all the plots of the paper submited for publishing in the M6 Special Issue #####
######################################################################################################################

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.formula.api as smf
import seaborn as sns
from statistics import stdev
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# set current working directory
os.chdir("C:/Users/TasosK/Desktop/Tasos/A-Research/M6 Discussion Paper/Submission_Workspace")


def IR_calculation(hist_data, submission, IR_flag=True):

    # Function for computing IR
    # given the historical data of the M6 assets 
    # and the submission made by the team 

    # It is a variation of the function uploaded to the M6-methods repo
    # https://github.com/Mcompetitions/M6-methods/blob/main/RPS%20and%20IR%20calculation.py
    # without filling the missing values of the assets within the function

    asset_id = pd.unique(hist_data.symbol)
    asset_id = sorted(asset_id) 

    #Compute percentage returns
    returns = pd.DataFrame(columns = ["ID", "Return"])

    #Investment weights
    weights = submission[["ID","Decision"]]

    RET = []

    for i in range(len(asset_id)):
        temp = hist_data.loc[hist_data.symbol==asset_id[i]]
        temp = temp.sort_values(by='date')
        temp.reset_index(inplace=True, drop=True)       
        RET.append(temp.price.pct_change()*weights.loc[weights.ID==asset_id[i]].Decision.values[0])
    
    ret = np.log(1+pd.DataFrame(RET).sum()[1:])
    sum_ret = sum(ret)
    sdp = stdev(ret)
    
    if IR_flag:
        # if IR
        output = {'IR' : sum_ret/sdp,
                'details' : list(ret)}
    else:
        # if returns
        output = {'IR' : sum_ret,
                'details' : list(ret)}
    
    return output


############################## DATA IMPORTS ##############################

# import data
sub_path = "./Data/submissions.csv"
gir_path = "./Data/All_teams_Global_IR.xlsx"
dret_per_team_path = "./Data/Daily_returns.xlsx"
ir_p_ev_per_path = "./Data/All_teams_monthly_IR.xlsx"
asset_data_path = "./Data/assets_m6.csv"
strategy_changes_path = "./Data/strategy_changes_data.csv"
asset_types_path = "./Data/stocks_etfs.xlsx"

submissions = pd.read_csv(sub_path)
global_IR_per_team = pd.read_excel(gir_path, index_col= 0)
daily_returns_per_team = pd.read_excel(dret_per_team_path, index_col= 0)
per_eval_period_IR = pd.read_excel(ir_p_ev_per_path, index_col= 0)
asset_data = pd.read_csv(asset_data_path)
strategy_changes = pd.read_csv(strategy_changes_path, index_col=1)
types = pd.read_excel(asset_types_path, index_col=0)

submissions = submissions[submissions.IsActive==1]
daily_portfolio_value_per_team = daily_returns_per_team.cumsum(axis = 1)
per_eval_period_IR = per_eval_period_IR.iloc[:,-12:]
per_eval_period_IR = per_eval_period_IR.dropna()
asset_data.date = pd.to_datetime(asset_data.date)

##########################################################################

############################## BASIC PRE-PROCESSING ##############################

# rename the benchmark submission in the data
per_eval_period_IR = per_eval_period_IR.rename(index={"32cdcc24": "Benchmark"})
global_IR_per_team = global_IR_per_team.rename(index={"32cdcc24": "Benchmark"})
daily_returns_per_team = daily_returns_per_team.rename(index={"32cdcc24": "Benchmark"})
daily_portfolio_value_per_team = daily_portfolio_value_per_team.rename(index={"32cdcc24": "Benchmark"})
submissions.loc[submissions.Team=="32cdcc24", "Team"] = "Benchmark"

# remove all submissions that have the exact same weights as the benchmark
Active_teams_IR_per_eval_period = per_eval_period_IR.loc[(per_eval_period_IR == per_eval_period_IR.loc["Benchmark"]).all(axis=1)==False]
all_active_ir_teams = list(Active_teams_IR_per_eval_period.index)
all_active_ir_teams.append(per_eval_period_IR.index[0])

# rank the active teams based on their global IR
global_IR_per_team = global_IR_per_team.loc[all_active_ir_teams]
global_IR_per_team = global_IR_per_team.sort_values(by='Global', ascending=False)

# sort all data on the active teams and by their global IR
Active_teams_IR_per_eval_period = per_eval_period_IR.loc[global_IR_per_team.index]
daily_returns_per_team = daily_returns_per_team.loc[global_IR_per_team.index]
daily_portfolio_value_per_team = daily_portfolio_value_per_team.loc[global_IR_per_team.index]

Active_teams_IR_per_eval_period["Global IR"] = global_IR_per_team["Global"]
Active_teams_IR_per_eval_period["Average Win"] = Active_teams_IR_per_eval_period.iloc[:,:12].mean(axis=1) # mean monthly IR of each team

active_teams_list = [x for x in Active_teams_IR_per_eval_period.index if x in ["Benchmark"] + list(pd.unique(submissions.Team))]

Active_teams_IR_per_eval_period = Active_teams_IR_per_eval_period.loc[active_teams_list]
Active_teams_IR_per_eval_period = Active_teams_IR_per_eval_period.sort_values(by='Global IR', ascending=False)

Active_teams_IR_per_eval_period["Monthly_STD"] = Active_teams_IR_per_eval_period.iloc[:,0:12].std(axis=1)

Active_teams_IR_per_eval_period["Unique_Submissions"] = 0

for team_id in Active_teams_IR_per_eval_period.index:

    team_submissions = submissions.loc[submissions.Team==team_id]
    Active_teams_IR_per_eval_period.loc[team_id, "Unique_Submissions"] = len([x for x in pd.unique(team_submissions.Submission) if x != "Trial run"])

Active_teams_IR_per_eval_period["Rank"] = [x+1 for x in list(range(Active_teams_IR_per_eval_period.shape[0]))]

##################################################################################

# AII calculation
Active_teams_IR_per_eval_period["AII"] = 0
for team_id in Active_teams_IR_per_eval_period.index:
    if team_id!="Benchmark":
        team_submissions = submissions.loc[submissions.Team==team_id]
        sep_submissions = pd.DataFrame([x for x in pd.unique(team_submissions.Submission) if x != "Trial run"]).replace({"1st Submission": 1, "2nd Submission": 2, "3rd Submission": 3,
                                                                                                  "4th Submission": 4, "5th Submission": 5, "6th Submission": 6,
                                                                                                  "7th Submission": 7, "8th Submission": 8, "9th Submission": 9,
                                                                                                  "10th Submission": 10, "11th Submission": 11, "12th Submission": 12,})
        sub_vector = np.array([0*x for x in list(range(12))])
        sub_vector[list(sep_submissions.iloc[:,0].values -1)] = 1

        y = []
        counter = 0
        for tmp in range(len(sub_vector)-1):
            if sub_vector[tmp+1]==0:
                counter+=1
            else:
                y.append(counter)
                counter = 0
        if sub_vector[tmp+1]==0:
            y.append(counter)
        y = [x**2 for x in y]

        Active_teams_IR_per_eval_period.loc[team_id, "AII"] = np.mean(y)**0.5

# Output snapshot of the information of the assets for the 'alternate_strategies_exploration.py' script
df_general_info = Active_teams_IR_per_eval_period.copy()
df_general_info.to_csv("./Data/Df_general_info.csv")


# create groups of teams based on AII
temp_list = {}
temp_list_ir = {}

for team_adi in pd.unique(Active_teams_IR_per_eval_period["AII"]):
    
    temp_list_ir[team_adi] = list(Active_teams_IR_per_eval_period.loc[Active_teams_IR_per_eval_period["AII"]==team_adi, "Global IR"].values)

temp_df_dict = {}
for dict_ind in range(len(temp_list_ir.keys())):
    temp_df_dict['{}'.format(list(temp_list_ir.keys())[dict_ind])] = pd.Series(temp_list_ir[list(temp_list_ir.keys())[dict_ind]])
temp_df = pd.DataFrame(temp_df_dict)


temp_df = temp_df.dropna(axis=1, how='all')
temp_df = temp_df[temp_df.columns.sort_values()]
temp_df.columns = [float(x) for x in temp_df.columns]

temp_df_groupped = pd.DataFrame(index = list(range(150)), columns = ["0.0", "0.1-0.5", "0.51-1.0", "1.1-1.5", "1.51-2.0", "2.1-3.0", "3.1-4.0", "4.1-6.0", "6.0-11.0"])

temp_list=temp_df[0].dropna().values.tolist()

exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["0.0"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>0 and x<=0.5]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["0.1-0.5"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>0.5 and x<=1]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["0.51-1.0"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>1 and x<=1.5]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["1.1-1.5"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>1.5 and x<=2]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["1.51-2.0"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>2 and x<=3]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["2.1-3.0"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>3 and x<=4]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["3.1-4.0"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>4 and x<=6]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["4.1-6.0"] = temp_list

temp_list=[]
for ind in [x for x in temp_df.columns if x>6]:
    temp_list = temp_list+temp_df[ind].dropna().values.tolist()
exceeding_length = 150-len(temp_list)
temp_list = temp_list+(list([np.nan]*exceeding_length))
temp_df_groupped["6.0-11.0"] = temp_list

temp_df_groupped = temp_df_groupped.dropna(axis = 0, how = 'all')
group_dfs_1 = temp_df_groupped.copy()

swarm_df_temp = pd.DataFrame(index=[0], columns=["range", "IR"])
c_ind = 0
for x_ind in range(temp_df_groupped.shape[0]):
    for y_ind in range(temp_df_groupped.shape[1]):
        swarm_df_temp.loc[c_ind, "range"] = temp_df_groupped.columns[y_ind]
        swarm_df_temp.loc[c_ind, "IR"] = temp_df_groupped.iloc[x_ind, y_ind]
        c_ind+=1
swarm_df_temp = swarm_df_temp.dropna()

plt.cla()
plt.clf()
a4_dims = (12, 9)
plt.figure(figsize=a4_dims)
ax = sns.swarmplot(data=swarm_df_temp, x="range", y="IR", legend=False, size=10)
ax = sns.boxplot(data=temp_df_groupped, color="#172c69", medianprops={"color": "lightgrey", "linewidth": 1.5}, fill=False)
plt.grid()
plt.xlabel('AII', fontsize=22)
plt.ylabel('IR', fontsize=22)
plt.yticks(fontsize=18)
plt.xticks(fontsize=16, rotation=45)
plt.tight_layout()
plt.savefig("./Results/SV_SNS_IR_vs_AII_boxplot_all_grouped.eps", format='eps') 
plt.savefig("./Results/SV_SNS_IR_vs_AII_boxplot_all_grouped.png") 



temp_list = {}
temp_list_ir = {}
for unique_sub in pd.unique(Active_teams_IR_per_eval_period["Unique_Submissions"]):
    temp_list_ir[unique_sub] = list(Active_teams_IR_per_eval_period.loc[Active_teams_IR_per_eval_period["Unique_Submissions"]==unique_sub, "Global IR"].values)

temp_df = pd.DataFrame({'1': pd.Series(temp_list_ir[1]), '2': pd.Series(temp_list_ir[2]), '3': pd.Series(temp_list_ir[3]), 
                        '4': pd.Series(temp_list_ir[4]), '5': pd.Series(temp_list_ir[5]), '6': pd.Series(temp_list_ir[6]),
                        '7': pd.Series(temp_list_ir[7]), '8': pd.Series(temp_list_ir[8]), '9': pd.Series(temp_list_ir[9]), 
                        '10': pd.Series(temp_list_ir[10]), '11': pd.Series(temp_list_ir[11]), '12': pd.Series(temp_list_ir[12])})

group_dfs_3 = temp_df.copy()
swarm_df_temp = pd.DataFrame(index=[0], columns=["range", "IR"])
c_ind = 0
for x_ind in range(temp_df.shape[0]):
    for y_ind in range(temp_df.shape[1]):
        swarm_df_temp.loc[c_ind, "range"] = temp_df.columns[y_ind]
        swarm_df_temp.loc[c_ind, "IR"] = temp_df.iloc[x_ind, y_ind]
        c_ind+=1
swarm_df_temp = swarm_df_temp.dropna()

mpl.rcParams.update(mpl.rcParamsDefault)
plt.cla()
plt.clf()
ax = sns.swarmplot(data=swarm_df_temp, x="range", y="IR", legend=False, size=10)
g=sns.boxplot(temp_df, color="#172c69", medianprops={"color": "lightgrey", "linewidth": 1.5}, fill=False)
plt.grid()
plt.gca().set_xticklabels([str(x) for x in list(range(1,13))])
g.set_xlabel('Unique Submissions', fontsize=25)
g.set_ylabel('IR', fontsize=25)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.tight_layout()
plt.savefig("./Results/SV_SNS_IR_vs_Unique_submissions.eps", format='eps')
plt.savefig("./Results/SV_SNS_IR_vs_Unique_submissions.png") 

strategy_changes = strategy_changes.iloc[:,1:]
strategy_changes.loc["Benchmark", "strategies"] = 1
strategy_changes.loc["Benchmark", "changes"] = 0
strategy_changes.loc["Benchmark", "sig_changes"] = 0
strategy_changes.loc["Benchmark", "IR"] = Active_teams_IR_per_eval_period.loc["Benchmark", "Global IR"]

temp_df = Active_teams_IR_per_eval_period.copy()
temp_df = temp_df.iloc[:,:13]
temp_df["Strategy Changes"] = strategy_changes.changes
temp_df["Total Strategies"] = strategy_changes.strategies
# set strategy info for benchmark
temp_df.loc["08986844", "Strategy Changes"] = 0
temp_df.loc["08986844", "Total Strategies"] = 1

temp_list = {}
temp_list_tot_strat = {}

for str_cntr in np.sort(pd.unique(temp_df["Strategy Changes"])):
    temp_list[str_cntr] = list(temp_df.loc[temp_df["Strategy Changes"]==str_cntr, "Global IR"].values)

for str_cntr in np.sort(pd.unique(temp_df["Total Strategies"])):
    temp_list_tot_strat[str_cntr] = list(temp_df.loc[temp_df["Total Strategies"]==str_cntr, "Global IR"].values)

temp_df_str_chng_dict = {}
for dict_ind in range(len(temp_list.keys())):
    temp_df_str_chng_dict['{}'.format(list(temp_list.keys())[dict_ind])] = pd.Series(temp_list[list(temp_list.keys())[dict_ind]]).sort_values(ascending=True)
temp_df_str_chng = pd.DataFrame(temp_df_str_chng_dict)


temp_swarm_df = temp_df[["Global IR", "Strategy Changes"]]
temp_swarm_df["# Submissions"] = Active_teams_IR_per_eval_period["Unique_Submissions"]
temp_swarm_df.loc[temp_swarm_df["# Submissions"]==0, "# Submissions"]=1
group_dfs_2 = temp_swarm_df.copy()

plt.cla()
plt.clf()
a4_dims = (12, 9)
plt.figure(figsize=a4_dims)
palette = sns.color_palette("Blues",n_colors=12)
ax = sns.swarmplot(data=temp_swarm_df, x="Strategy Changes", y="Global IR", legend=False, size=10)
# Add legend to top right, outside plot region
sns.boxplot(data=temp_df_str_chng, color="#172c69", medianprops={"color": "grey", "linewidth": 1.5}, fill=False)
plt.grid()
plt.xlabel('Strategy Changes', fontsize=22)
plt.ylabel('IR', fontsize=22)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.gca().set_xticklabels([0,1,2,3,4,5,6,7,8])
plt.tight_layout()
plt.savefig("./Results/SV_SNS_IR_vs_strategy_changes_boxplot.eps", format='eps')
plt.savefig("./Results/SV_SNS_IR_vs_strategy_changes_boxplot.png") 

temp = Active_teams_IR_per_eval_period.sort_values(by="Global IR", ascending=False)
first_team_sub = temp.iloc[0,]; second_team_sub = temp.iloc[1,]; third_team_sub = temp.iloc[2,]; fourth_team_sub = temp.iloc[3,]; fifth_team_sub = temp.iloc[4,]
sixth_team_sub = temp.iloc[5,]; seventh_team_sub = temp.iloc[6,]; eigthth_team_sub = temp.iloc[7,]; ninth_team_sub = temp.iloc[8,]; tenth_team_sub = temp.iloc[9,]

first_team_name = first_team_sub.name; second_team_name = second_team_sub.name; third_team_name = third_team_sub.name; fourth_team_name = fourth_team_sub.name; fifth_team_name = fifth_team_sub.name
sixth_team_name = sixth_team_sub.name; seventh_team_name = seventh_team_sub.name; eightth_team_name = eigthth_team_sub.name; ninth_team_name = ninth_team_sub.name; tenth_team_name = tenth_team_sub.name

asset_id = pd.unique(asset_data.symbol)

for i in range(len(pd.unique(asset_data.date))):
    if len(asset_data[asset_data.date == pd.unique(asset_data.date)[i]])<len(asset_id):
        for asset in [x for x in asset_id if x not in asset_data[asset_data.date == pd.unique(asset_data.date)[i]].symbol.values]:
            right_price = asset_data[asset_data.symbol==asset].sort_values(by='date')
            right_price = right_price[right_price.date <= pd.unique(asset_data.date)[i]]
            right_price = right_price.price.iloc[-1]
            new_row_for_insertion = pd.DataFrame([asset, pd.unique(asset_data.date)[i],right_price]).T
            new_row_for_insertion.columns = ["date", "symbol", "price"]
            new_index = asset_data.index[-1] +1
            asset_data.loc[new_index] = new_row_for_insertion.values[0]

wanted_asset = asset_data.loc[asset_data.symbol=='IVV', ['date', 'price']]
wanted_asset = wanted_asset.sort_values(by="date")
wanted_asset.price = wanted_asset.price.pct_change()
wanted_asset = wanted_asset.set_index("date") 

temp_plot_df = daily_returns_per_team.loc[[first_team_name, second_team_name, third_team_name,
                    fourth_team_name, fifth_team_name, sixth_team_name,
                    seventh_team_name, eightth_team_name, ninth_team_name,
                    tenth_team_name, "Benchmark"]]

temp_plot_df = temp_plot_df.T
wanted_asset = wanted_asset[wanted_asset.index.isin(temp_plot_df.index)]
temp_plot_df["IVV"] = wanted_asset.price
temp_plot_df = temp_plot_df.cumsum(axis=0)
temp_plot_df = temp_plot_df+1
temp_plot_df.columns = [1,2,3,4,5,6,7,8,9,10, "Benchmark", "IVV"]
palette = sns.color_palette("Blues",n_colors=10)
palette.reverse()
palette.append("gray")
palette.append("darkgrey")

plt.cla()
plt.clf()
g=sns.relplot(temp_plot_df, kind='line',
              facet_kws={'despine':False}, palette = palette,
              dashes=False, height=5, aspect=1.33)
plt.gca().lines[10].set_linestyle("--")
plt.gca().lines[11].set_linestyle("--")
plt.ylabel('Cumulative Return')
plt.grid()
plt.savefig("./Results/SV_SNS_Cumulative_Returns_top_10_Bench_n_IVV.eps", format='eps')
plt.savefig("./Results/SV_SNS_Cumulative_Returns_top_10_Bench_n_IVV.png") 

# plot average ir vs global ir along with regression lines
temporary_df = pd.DataFrame(Active_teams_IR_per_eval_period["Global IR"])
temporary_df["Average_IR"] = Active_teams_IR_per_eval_period["Average Win"]
temporary_df.columns = ["Global", "Average_IR"]

model_95 = smf.quantreg('Average_IR ~ Global', temporary_df).fit(q=0.95)
model_50 = smf.quantreg('Average_IR ~ Global', temporary_df).fit(q=0.50)
model_05 = smf.quantreg('Average_IR ~ Global', temporary_df).fit(q=0.05)

# get y values
get_y_095 = lambda a, b: a + b * temporary_df.Global
y_095 = get_y_095(model_95.params['Intercept'], model_95.params['Global'])

get_y_050 = lambda a, b: a + b * temporary_df.Global
y_050 = get_y_050(model_50.params['Intercept'], model_50.params['Global'])

get_y_005 = lambda a, b: a + b * temporary_df.Global
y_005 = get_y_005(model_05.params['Intercept'], model_05.params['Global'])

temporary_df["y005"] = y_005
temporary_df["y050"] = y_050
temporary_df["y095"] = y_095

plt.cla()
plt.clf()
g = sns.relplot(data=temporary_df, kind='scatter', x='Global', y='Average_IR', color="#172c69", facet_kws={'despine':False}, height=5, aspect= 1.33)
g.map(sns.lineplot, 'Global', 'y005', color="#172c69")
g.map(sns.lineplot, 'Global', 'y095', color="#172c69")
plt.grid()
g.set_xlabels('IR', fontsize=14)
g.set_ylabels('Average Monthly IR', fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.tight_layout()
g.savefig('./Results/SV_SNS_Average_IR_VS_Global_IR.eps', format='eps')
g.savefig("./Results/SV_SNS_Average_IR_VS_Global_IR.png") 

# regression to get the r and p values for the regression between monthly and global IR
# They are shown in the manuscript in Figure 5 caption
print("********************************************")
print("****** Global vs Monthly IR Regression ******")
print("********************************************")
X2 = sm.add_constant(temporary_df["Global"].astype(float))
est = sm.OLS(temporary_df["Average_IR"].astype(float), X2)
est2 = est.fit()
print(est2.summary())

bot_10_pct = np.quantile(Active_teams_IR_per_eval_period.iloc[:,:12], 0.1)
top_10_pct = np.quantile(Active_teams_IR_per_eval_period.iloc[:,:12], 0.9)
plt.cla()
plt.clf()
g=sns.boxplot(Active_teams_IR_per_eval_period.iloc[:10,:12].T, color="#172c69", medianprops={"color": "grey", "linewidth": 1.5}, fill=False)
sns.stripplot(Active_teams_IR_per_eval_period.iloc[:10,:12].T, color="#172c69")
plt.grid()
g.axhline(y=bot_10_pct, color="#172c69", linestyle='dotted')
g.axhline(y=top_10_pct, color="#172c69", linestyle='dotted')
plt.gca().set_xticklabels([str(x) for x in list(range(1,11))])
g.set_xlabel('Rank', fontsize=14)
g.set_ylabel('IR', fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig("./Results/SV_SNS_Monthly_IR_Boxplots_for_top_10_teams.eps", format='eps')
plt.savefig("./Results/SV_SNS_Monthly_IR_Boxplots_for_top_10_teams.png") 



temp = Active_teams_IR_per_eval_period.copy()
temp.iloc[:,:12] = 160 + 1 - temp.iloc[:,:12].rank()
top_15_teams_submissions = pd.DataFrame(0, index=list(range(1,16)), columns=list(range(1,13)))
top_team_cntr = 0

for team_id in Active_teams_IR_per_eval_period.index:
    team_submissions = submissions.loc[submissions.Team==team_id]

    if top_team_cntr<15:
        sep_submissions = pd.DataFrame([x for x in pd.unique(team_submissions.Submission) if x != "Trial run"]).replace({"1st Submission": 1,
                                                                                                  "2nd Submission": 2,
                                                                                                  "3rd Submission": 3,
                                                                                                  "4th Submission": 4,
                                                                                                  "5th Submission": 5,
                                                                                                  "6th Submission": 6,
                                                                                                  "7th Submission": 7,
                                                                                                  "8th Submission": 8,
                                                                                                  "9th Submission": 9,
                                                                                                  "10th Submission": 10,
                                                                                                  "11th Submission": 11,
                                                                                                  "12th Submission": 12,})
        
        top_15_teams_submissions.iloc[top_team_cntr,[x-1 for x in sep_submissions.values]] = 1
    top_team_cntr+=1

top_15_teams_submissions.index = [str(x) for x in top_15_teams_submissions.index]
top_15_teams_submissions.columns = [str(x) for x in top_15_teams_submissions.columns]
 
plt.cla()
plt.clf()
a4_dims = (8, 8)
plt.figure(figsize=a4_dims)
g = sns.heatmap(top_15_teams_submissions, cmap = 'Blues', linewidth=.3, linecolor='lightgrey', cbar=False)
g.tick_params(left=False, bottom=False) ## other options are right and top
g.set_xlabel('Submission Period', fontsize=16)
g.set_ylabel('Rank', fontsize=16)
plt.yticks(fontsize=12, rotation=90)
plt.xticks(fontsize=12)

plt.savefig("./Results/SV_SNS_Rank_vs_Submissions.eps", format='eps')
plt.savefig("./Results/SV_SNS_Rank_vs_Submissions.png")



One_sub_teams = Active_teams_IR_per_eval_period.loc[Active_teams_IR_per_eval_period.Unique_Submissions==1].iloc[:,:12]
Twelve_sub_teams = Active_teams_IR_per_eval_period.loc[Active_teams_IR_per_eval_period.Unique_Submissions==12].iloc[:,:12]
One_sub_teams = One_sub_teams.drop('Benchmark')

for team_id in One_sub_teams.index:
    for sub_ind in One_sub_teams.columns:
        if One_sub_teams.loc[team_id, sub_ind]>Active_teams_IR_per_eval_period.loc["Benchmark", sub_ind]:
            One_sub_teams.loc[team_id, sub_ind]=1
        else:
            One_sub_teams.loc[team_id, sub_ind]=0


for team_id in Twelve_sub_teams.index:
    for sub_ind in Twelve_sub_teams.columns:
        if Twelve_sub_teams.loc[team_id, sub_ind]>Active_teams_IR_per_eval_period.loc["Benchmark", sub_ind]:
            Twelve_sub_teams.loc[team_id, sub_ind]=1
        else:
            Twelve_sub_teams.loc[team_id, sub_ind]=0

temp_plot_df = pd.DataFrame(One_sub_teams.sum()/One_sub_teams.shape[0])
temp_plot_df.columns = ["1"]
temp_plot_df["12"] = Twelve_sub_teams.sum()/Twelve_sub_teams.shape[0]
temp_plot_df.index = list(range(1,13))
temp_plot_df = temp_plot_df*100
palette = sns.color_palette("Blues",n_colors=2)

temp_bench_irs = pd.DataFrame(index = list(range(1,13)), columns = ["IR"])
temp_bench_irs.IR = Active_teams_IR_per_eval_period.loc["Benchmark"][:12].values
temp_bench_irs.reset_index(inplace=True)
temp_bench_irs.columns = ["index", "IR"]


plt.cla()
plt.clf()
sns.relplot(temp_plot_df, kind='line',
              facet_kws={'despine':False}, palette = palette,
              dashes=False, height=5, aspect= 1.33)

g=sns.relplot(temp_plot_df, kind='line',
              facet_kws={'despine':False}, palette = palette,
              dashes=False, height=5, aspect= 1.33)
g.legend.get_title().set_fontsize(14)
g._legend.set_title("# Submissions")
g._legend.set_bbox_to_anchor([1.25,0.5])
plt.setp(g._legend.get_texts(), fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel('Submission Period', fontsize=14)
plt.ylabel('% Teams', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()

ax2 = plt.twinx()
sns.scatterplot(data=temp_bench_irs, x='index', y='IR', ax=ax2, color='#7d4d6c')#fe7201
ax2.set_ylabel('IR', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("./Results/SV_SNS_1_vs_12_sub_pct_over_bench.eps", bbox_inches='tight', format='eps')
plt.savefig("./Results/SV_SNS_1_vs_12_sub_pct_over_bench.png", bbox_inches='tight')

teams_df = pd.DataFrame(index = pd.unique(submissions.Team), columns=["Stocks_pct", "ETFs_pct"])

for team_id in pd.unique(submissions.Team):

    team_subs = submissions.loc[submissions.Team==team_id, ]
    team_subs = team_subs.loc[team_subs.Submission!='Trial run']
    team_subs['Decision'] = [np.abs(x) for x in team_subs['Decision']]
    team_subs = team_subs.groupby(['Symbol'])['Decision'].agg(['sum'])
    team_subs["type"] = types["Type"]
    team_subs = team_subs.groupby(['type'])['sum'].agg(['sum'])
    
    teams_df.loc[team_id, "Stocks_pct"] = team_subs.loc["Stock", "sum"]
    teams_df.loc[team_id, "ETFs_pct"] = team_subs.loc["ETF", "sum"]

Active_teams_IR_per_eval_period["Stocks_pct"] = teams_df["Stocks_pct"]
Active_teams_IR_per_eval_period["Global Risk"] = daily_returns_per_team.std(axis=1)

temp = Active_teams_IR_per_eval_period.copy()
temp1 = temp[["Rank", "Stocks_pct"]]
temp1 = temp1.groupby(np.arange(len(temp1)) // 15).agg({'Rank': 'mean', 'Stocks_pct': 'mean'})
temp1 = temp1.astype('float64')
plt.cla()
plt.clf()
plt.grid()
ax = sns.regplot(data=temp1, x="Rank", y="Stocks_pct", ci=None, color="#172c69")
ax.set_ylabel("% Stocks")
ax.lines[0].set_linestyle("--")
plt.tight_layout()
plt.savefig('./Results/SV_SNS_Stock_pct_vs_Rank_per_n.eps', format='eps')
plt.savefig('./Results/SV_SNS_Stock_pct_vs_Rank_per_n.png')

# regression to get the r and p values for the regression between risk and rank
# They are shown in the manuscript in the caption of Figure 7a
print("********************************************")
print("******* Stock pct vs Rank Regression *******")
print("********************************************")
X2 = sm.add_constant(temp1["Rank"].astype(float))
est = sm.OLS(temp1["Stocks_pct"].astype(float), X2)
est2 = est.fit()
print(est2.summary())


temp1 = temp[["Rank", "Monthly_STD"]]
temp1 = temp1.groupby(np.arange(len(temp1)) // 15).agg({'Rank': 'mean', 'Monthly_STD': 'mean'})
plt.cla()
plt.clf()
ax = sns.regplot(data=temp1, x="Rank", y="Monthly_STD", ci=None, color="#172c69")
ax.lines[0].set_linestyle("--")
ax.set_xlabel('Rank', fontsize=14)
ax.set_ylabel('IR STD', fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('./Results/SV_SNS_sdMIR_vs_Rank_per_n.eps', format='eps')
plt.savefig('./Results/SV_SNS_sdMIR_vs_Rank_per_n.png')


temp1 = temp[["Rank", "Global Risk"]]
temp1 = temp1.groupby(np.arange(len(temp1)) // 15).agg({'Rank': 'mean', 'Global Risk': 'mean'})
plt.cla()
plt.clf()
plt.grid()
ax = sns.regplot(data=temp1, x="Rank", y="Global Risk", ci=None, color="#172c69")
ax.lines[0].set_linestyle("--")
ax.set_xlabel('Rank', fontsize=10)
ax.set_ylabel('Global Risk', fontsize=10)
plt.tight_layout()
plt.savefig('./Results/SV_SNS_global_risk_vs_Rank_per_n.eps', format='eps')
plt.savefig('./Results/SV_SNS_global_risk_vs_Rank_per_n.png')


# regression to get the r and p values for the regression between risk and rank
# They are shown in the manuscript in the caption of Figure 7a
print("********************************************")
print("********** Risk vs Rank Regression **********")
print("********************************************")
X2 = sm.add_constant(temp1["Rank"].astype(float))
est = sm.OLS(temp1["Global Risk"].astype(float), X2)
est2 = est.fit()
print(est2.summary())






















































