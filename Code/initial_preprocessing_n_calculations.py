import pandas as pd
import numpy as np
from statistics import stdev
import os
import warnings
warnings.filterwarnings('ignore')
import time
start = time.time()

def IR_modified_calculation(hist_data, submission, IR_flag=True):

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
        output = {'ret' : sum_ret,
                'details' : list(ret)}
    
    return output

# Read asset prices data (as provided by the M6 submission platform)
asset_data = pd.read_csv("./Data/assets_m6.csv")
asset_data.date = pd.to_datetime(asset_data.date)

# Read submissions file (similar to the template provided by the M6 submission platform)
all_submissions = pd.read_csv("./Data/submissions.csv")
all_submissions = all_submissions.iloc[:,:-3]
all_submissions = all_submissions[['Team', 'Submission', 'Evaluation', 'Symbol', 'Rank1',
       'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Decision']]
all_submissions = all_submissions.replace('META','FB') # replace the asset name to match in both files

# initialise the main ir file
ir_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = pd.unique(all_submissions.Evaluation))
ret_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = pd.unique(all_submissions.Evaluation))

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

cntr = 0

# iterate through all teams and calculate their IR for each (monthly) evaluation period
for team_id in ir_overview.index:

    teams_submissions = all_submissions.loc[all_submissions.Team==team_id]
    
    for eval_period in ir_overview.columns:
        
        period_team_sub = teams_submissions.loc[teams_submissions.Evaluation==eval_period]
        period_team_sub = period_team_sub.iloc[:,3:]
        period_team_sub.rename(columns = {'Symbol':'ID'}, inplace = True)

        period_asset_data = asset_data[((asset_data.date<=time_dividers.loc[eval_period, "End"]) & (asset_data.date>=time_dividers.loc[eval_period, "Start"]))]
        
        try:
            ir_overview.loc[team_id, eval_period] = IR_modified_calculation(hist_data = period_asset_data, submission = period_team_sub, IR_flag=True)['IR']
            ret_overview.loc[team_id, eval_period] = IR_modified_calculation(hist_data = period_asset_data, submission = period_team_sub, IR_flag=False)['ret']
            
        except:
            print("Team ", team_id, "has no data for evaluation period ", eval_period)

    cntr+=1
    
    print("Calculated for team with id: ", team_id, ". Teams finished: ", cntr, "/", len(ir_overview.index))

# # export to seperate file
ir_overview.to_excel("./Data/All_teams_monthly_IR.xlsx")
# fix submission for team bc4b0314 which is irregular for the last 3 months
ret_overview[:40] = ret_overview[:40].fillna(0)
ret_overview.to_excel("./Data/All_teams_monthly_Returns.xlsx")

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

# Calculate Quarterly IR & returns
ret_quarterly_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = ["q1", "q2", "q3", "q4"])
ir_quarterly_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = ["q1", "q2", "q3", "q4"])
asset_id = pd.unique(asset_data.symbol)
asset_id = sorted(asset_id) 

for team_id in ret_quarterly_overview.index:
    
    teams_submissions = all_submissions.loc[all_submissions.Team==team_id]
    
    for q_id in ["q1", "q2", "q3", "q4"]:
        try:
            quarter_id_info = quarter_info.loc[q_id]
            quarterly_daily_returns = []
            
            for eval_id in list(quarter_id_info.values):
    
                period_asset_data = asset_data[((asset_data.date<=time_dividers.loc[eval_id, "End"]) & (asset_data.date>=time_dividers.loc[eval_id, "Start"]))]
            
                period_team_sub = teams_submissions.loc[teams_submissions.Evaluation==eval_id]
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

            ret_quarterly_overview.loc[team_id, q_id] = sum_ret
            ir_quarterly_overview.loc[team_id, q_id] = sum_ret/sdp
        except:
            print("Team ", team_id, "has no data for evaluation period ", eval_id)

# export to seperate file
ret_quarterly_overview.to_excel("./Data/All_teams_Quarterly_Returns.xlsx")
ir_quarterly_overview.to_excel("./Data/All_teams_Quarterly_IR.xlsx")


# Calculate Global Returns 
ret_global_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = ["Global"])
ir_global_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = ["Global"])
risk_global_overview = pd.DataFrame(index=pd.unique(all_submissions.Team), columns = ["Global"])
asset_id = pd.unique(asset_data.symbol)
asset_id = sorted(asset_id) 

for team_id in ret_global_overview.index:
    
    teams_submissions = all_submissions.loc[all_submissions.Team==team_id]
    try:
        quarterly_daily_returns = []
        for q_id in ["q1", "q2", "q3", "q4"]:
            
            quarter_id_info = quarter_info.loc[q_id]
            
            
            for eval_id in list(quarter_id_info.values):

                period_asset_data = asset_data[((asset_data.date<=time_dividers.loc[eval_id, "End"]) & (asset_data.date>=time_dividers.loc[eval_id, "Start"]))]
                
                period_team_sub = teams_submissions.loc[teams_submissions.Evaluation==eval_id]
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
        
        ret_global_overview.loc[team_id, "Global"] = sum_ret
        ir_global_overview.loc[team_id, "Global"] = sum_ret/sdp
        risk_global_overview.loc[team_id, "Global"] = sdp
        
    except:
        print("Team ", team_id, "has no data for evaluation period ", eval_id)

# export to seperate file
ret_global_overview.to_excel("./Data/All_teams_Global_Returns.xlsx")
ir_global_overview.to_excel("./Data/All_teams_Global_IR.xlsx")
risk_global_overview.to_excel("./Data/All_teams_Global_Risk.xlsx")


##########################################################################################################################################################
# Finished with the IR calculations
# Calculate the daily returns for each team
##########################################################################################################################################################

# Calculate daily returns for each team
all_dates = pd.unique(asset_data.date)
all_dates = all_dates[all_dates>=time_dividers.iloc[0,1]]
all_dates = all_dates[all_dates<=time_dividers.iloc[12,1]]

daily_returns_per_team = pd.DataFrame(index = pd.unique(all_submissions.Team), columns = all_dates[1:])

for team_id in daily_returns_per_team.index:
    
    teams_submissions = all_submissions.loc[all_submissions.Team==team_id]
    
    quarterly_daily_returns = []
    for q_id in ["q1", "q2", "q3", "q4"]:

        quarter_id_info = quarter_info.loc[q_id]
        
        for eval_id in list(quarter_id_info.values):
            try:
                period_asset_data = asset_data[((asset_data.date<=time_dividers.loc[eval_id, "End"]) & (asset_data.date>=time_dividers.loc[eval_id, "Start"]))]
                
                period_team_sub = teams_submissions.loc[teams_submissions.Evaluation==eval_id]
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
            except:
                print("Team ", team_id, "has no data for evaluation period ", eval_id)
    
    daily_returns_per_team.loc[team_id][-len(quarterly_daily_returns):] = quarterly_daily_returns

daily_returns_per_team.to_excel("./Data/Daily_returns.xlsx")

# 867 seconds
# ~14 minutes
end = time.time()
print(end - start)




