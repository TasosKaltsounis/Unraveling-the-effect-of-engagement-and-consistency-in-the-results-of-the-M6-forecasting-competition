import pandas as pd
import numpy as np
from statistics import stdev
import os
import warnings
warnings.filterwarnings('ignore')


############################## DATA IMPORTS ##############################

asset_data_path = "./Data/assets_m6.csv"
sub_path = "./Data/submissions.csv"

# Read asset prices data (as provided by the M6 submission platform)
asset_data = pd.read_csv(asset_data_path)
# Read submissions file (similar to the template provided by the M6 submission platform)
all_submissions = pd.read_csv(sub_path)

##########################################################################

asset_data.date = pd.to_datetime(asset_data.date)
all_submissions = all_submissions.replace('META','FB') # replace the asset name to match in both files
all_submissions = all_submissions[all_submissions.IsActive==1]
all_submissions = all_submissions[['Team', 'Submission', 'Evaluation', 'Symbol', 'Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Decision']]

# initialise the main ir file
ir_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = pd.unique(all_submissions.Evaluation))

# fill the missing asset values
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

# form a df with the evaluation periods of the M6 Forecasting Competition
time_dividers = pd.DataFrame(index = pd.unique(all_submissions.Evaluation), columns = ["Start", "End"])

time_dividers.iloc[0, 0] = pd.Timestamp(2022, 2, 4) # year, month, day
time_dividers.iloc[0, 1] = pd.Timestamp(2022, 3, 4) 

time_dividers.iloc[1, 0] = pd.Timestamp(2022, 3, 4) 
time_dividers.iloc[1, 1] = pd.Timestamp(2022, 4, 1)

time_dividers.iloc[2, 0] = pd.Timestamp(2022, 4, 1) 
time_dividers.iloc[2, 1] = pd.Timestamp(2022, 4, 29) 

time_dividers.iloc[3, 0] = pd.Timestamp(2022, 4, 29) 
time_dividers.iloc[3, 1] = pd.Timestamp(2022, 5, 27) 

time_dividers.iloc[4, 0] = pd.Timestamp(2022, 5, 27) 
time_dividers.iloc[4, 1] = pd.Timestamp(2022, 6, 24) 

time_dividers.iloc[5, 0] = pd.Timestamp(2022, 6, 24) 
time_dividers.iloc[5, 1] = pd.Timestamp(2022, 7, 22) 

time_dividers.iloc[6, 0] = pd.Timestamp(2022, 7, 22) 
time_dividers.iloc[6, 1] = pd.Timestamp(2022, 8, 19) 

time_dividers.iloc[7, 0] = pd.Timestamp(2022, 8, 19) 
time_dividers.iloc[7, 1] = pd.Timestamp(2022, 9, 16) 

time_dividers.iloc[8, 0] = pd.Timestamp(2022, 9, 16) 
time_dividers.iloc[8, 1] = pd.Timestamp(2022, 10, 14)

time_dividers.iloc[9, 0] = pd.Timestamp(2022, 10, 14) 
time_dividers.iloc[9, 1] = pd.Timestamp(2022, 11, 11) 

time_dividers.iloc[10, 0] = pd.Timestamp(2022, 11, 11) 
time_dividers.iloc[10, 1] = pd.Timestamp(2022, 12, 9) 

time_dividers.iloc[11, 0] = pd.Timestamp(2022, 12, 9)
time_dividers.iloc[11, 1] = pd.Timestamp(2023, 1, 6) 

time_dividers.iloc[12, 0] = pd.Timestamp(2023, 1, 6) 
time_dividers.iloc[12, 1] = pd.Timestamp(2023, 2, 3) 

# form a df with the quarter evaluation periods of the M6 Forecasting Competition
quarter_info = pd.DataFrame(index = ["q1", "q2", "q3", "q4"], columns = ["1st_month", "2nd_month", "3rd_month"])
quarter_info.loc["q1", "1st_month"] = pd.unique(all_submissions.Evaluation)[1]
quarter_info.loc["q1", "2nd_month"] = pd.unique(all_submissions.Evaluation)[2]
quarter_info.loc["q1", "3rd_month"] = pd.unique(all_submissions.Evaluation)[3]
quarter_info.loc["q2", "1st_month"] = pd.unique(all_submissions.Evaluation)[4]
quarter_info.loc["q2", "2nd_month"] = pd.unique(all_submissions.Evaluation)[5]
quarter_info.loc["q2", "3rd_month"] = pd.unique(all_submissions.Evaluation)[6]
quarter_info.loc["q3", "1st_month"] = pd.unique(all_submissions.Evaluation)[7]
quarter_info.loc["q3", "2nd_month"] = pd.unique(all_submissions.Evaluation)[8]
quarter_info.loc["q3", "3rd_month"] = pd.unique(all_submissions.Evaluation)[9]
quarter_info.loc["q4", "1st_month"] = pd.unique(all_submissions.Evaluation)[10]
quarter_info.loc["q4", "2nd_month"] = pd.unique(all_submissions.Evaluation)[11]
quarter_info.loc["q4", "3rd_month"] = pd.unique(all_submissions.Evaluation)[12]

# Calculate Global IR 
ir_global_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = ["Global"])
asset_id = pd.unique(asset_data.symbol)
asset_id = sorted(asset_id) 

team_id = ir_global_overview.index[3]
teams_submissions = all_submissions.loc[all_submissions.Team==team_id]

original_global_ir = pd.read_excel("./Data/All_teams_Global_IR.xlsx", index_col=0)
original_global_ir = original_global_ir.dropna(axis=0)
original_global_ir = original_global_ir.rename(index={"32cdcc24": "Benchmark"})


overview_df = pd.DataFrame(index = original_global_ir.index, columns = list(range(1,13)))

for team_id in ir_global_overview.index:

    teams_submissions = all_submissions.loc[all_submissions.Team==team_id]
    team_sub_periods = [x for x in pd.unique(teams_submissions.Submission) if x not in 'Trial run']
    total_count_of_team_subs = len(team_sub_periods)

    for submissions_cut in range(1,total_count_of_team_subs):

        if submissions_cut!=0:
            temp_team_sub_periods = team_sub_periods[:-submissions_cut]
            last_sub = temp_team_sub_periods[-1]
        else:
            temp_team_sub_periods = team_sub_periods
            last_sub = temp_team_sub_periods[-1]
        temp_teams_submissions = teams_submissions.copy()

        for missing_subs in [x for x in team_sub_periods if x not in temp_team_sub_periods]:
            for asset_index in pd.unique(temp_teams_submissions.Symbol):
                temp_teams_submissions.loc[((temp_teams_submissions.Submission == missing_subs) & (temp_teams_submissions.Symbol == asset_index)), "Decision"] = temp_teams_submissions.loc[((temp_teams_submissions.Submission == last_sub) & (temp_teams_submissions.Symbol == asset_index)), "Decision"].values[0]

        temp_teams_submissions.loc[~temp_teams_submissions.Submission.isin(temp_team_sub_periods), "Submission"] = last_sub

        try:
            quarterly_daily_returns = []
            for q_id in ["q1", "q2", "q3", "q4"]:
                
                quarter_id_info = quarter_info.loc[q_id]
                
                for eval_id in list(quarter_id_info.values):
                    
                    period_asset_data = asset_data[((asset_data.date<=time_dividers.loc[eval_id, "End"]) & (asset_data.date>=time_dividers.loc[eval_id, "Start"]))]
                    
                    period_team_sub = temp_teams_submissions.loc[temp_teams_submissions.Evaluation==eval_id]
                    period_team_sub = period_team_sub.iloc[:,3:]
                    period_team_sub.rename(columns = {'Symbol':'ID'}, inplace = True)
                    
                    #Compute percentage returns
                    returns = pd.DataFrame(columns = ["ID", "Return"])
                    
                    #Investment weights
                    weights = period_team_sub[["ID","Decision"]]
                    
                    RET = []
                    
                    for i in range(len(asset_id)):
                        temp = period_asset_data.loc[period_asset_data.symbol==asset_id[i]]
                        temp = temp.sort_values(by='date')
                        temp.reset_index(inplace=True, drop=True)       
                        RET.append(temp.price.pct_change()*weights.loc[weights.ID==asset_id[i]].Decision.values[0])
                    
                    ret = np.log(1+pd.DataFrame(RET).sum()[1:])
                    quarterly_daily_returns+=list(ret)

            sum_ret = sum(quarterly_daily_returns)
            sdp = stdev(quarterly_daily_returns)

            overview_df.loc[team_id, (total_count_of_team_subs-submissions_cut)] = sum_ret/sdp

        except:
            print(team_id)


overview_df["Global"] = original_global_ir["Global"]
overview_df.to_csv("./Data/Alternate_strategies_final_IR.csv")
