import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import time
start=time.time()
############################## DATA IMPORTS ##############################
# exported from the "alternate_strategies_file.py" script
overview_df_path = "./Data/Alternate_strategies_final_IR.csv"
# exported from the "paper_submission_code.py" script
df_general_info_path = "./Data/Df_general_info.csv"

strategy_changes_path = "./Data/strategy_changes_data.csv"

overview_df = pd.read_csv(overview_df_path, index_col=0)
df_general_info = pd.read_csv(df_general_info_path, index_col=0)
strategy_changes = pd.read_csv(strategy_changes_path, index_col=1)
##########################################################################


df_general_info["Total Strategies"] = strategy_changes.strategies
df_general_info.loc["08986844", "Total Strategies"] = 1

y = df_general_info["Rank"]
X = df_general_info[["Unique_Submissions", "AII", "Total Strategies"]]

# keep the top 25% of the teams and perform linear regression to evaluate the statistical importance of the three criteria
# The results are shown in Table 1 in the manuscript
print("********************************************")
print("*** Top 25% Regression to Three Criteria ***")
print("********************************************")
indexing = int(X.shape[0]*0.25)
X = X.iloc[:indexing,:]
y = y.iloc[:indexing]
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
print(est2.params)


end = time.time()
print(end - start)

# 1 second















































