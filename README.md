# Replication package for "Unraveling the effect of engagement and consistency in the results of the M6 forecasting competition", International Journal of Forecasting

Anastasios Kaltsounis, Evangelos Theodorou, Evangelos Spiliotis & Vassilios Assimakopoulos

## Overview 

The replication package consists of 6 scripts written in Python Programming Language. The scripts should be executed in the following order:

1. initial_preprocessing_n_calculations.py | Performs data pre-processing on the data provided by the organizers of the M6 forecasting competition. The files created are used in the next scripts for producing plots and results for the paper.
2. Main_Analysis_n_Plots.py | Calculates most of the results shown in the manuscript and outputs several plots that describe said results.
3. alternate_strategies_file.py | Creates a file with the different IR calculations for the "early stopping" strategy described in the manuscript. Since it is time-consuming, this analysis stands as a different script. The file produced is used by the 'alternate_strategies_analysis.py' script.
4. alternate_strategies_analysis.py | Produces the results of the "early stopping" strategy, outputs plots, and presents regression analysis results.
5. linear_regression_analysis.py | Applies the regression analysis discussed in the manuscript and presents the results of the respective models.
6. anova_analysis.py | Contains the template for the "one-way Anova tests" performed in the data set.


The Data used for this study were provided by the organizers of the M6 forecasting competition.

- submissions.csv | Contains all the submissions of all teams throughout the competition (https://github.com/Mcompetitions/M6-methods/tree/main/IJF%20paper).
- assets.csv | Contains all the assets used in the M6 forecasting Competition (https://github.com/Mcompetitions/M6-methods/tree/main).
- stocks_etfs.xlsx | Contains all the assets used in the M6 forecasting Competition with a flag as to whether they are stocks or ETFs.

Plots used in the manuscript are stored in the 'Results' folder. Linear regression model results are printed by the scripts.

All libraries used and their versions are shown in the requirements.txt file.

