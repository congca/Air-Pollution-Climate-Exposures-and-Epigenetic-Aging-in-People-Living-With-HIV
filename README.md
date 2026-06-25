# Air Pollution, Climate Exposures, and Epigenetic Aging in People Living With HIV

Code used for the analyses presented in the manuscript

> Air Pollution, Climate-Related Exposures, and Heterogeneity Across Epigenetic Aging Measures in People Living With HIV

## Files

Data_Preparation.py
Random_Forest_Feature_Selection.py
K_Means_and_DAG.py
Analysis.py


`Data_Preparation.py`
- data cleaning
- feature engineering
- preprocessing

`Random_Forest_Feature_Selection.py`
- random forest models
- feature importance ranking

`K_Means_and_DAG.py`
- K-means clustering
- PCA visualization
- DAG construction

`Analysis.py`
- Double Machine Learning
- XGBoost nuisance models
- treatment effect estimation
- exposure-response plots

## Data

The analytic dataset was derived from the Multicenter AIDS Cohort Study (MACS/MWCCS) and linked with environmental exposure data. Individual-level data are not publicly available because of data use restrictions.

## Software

Analyses were performed in Python using

- pandas
- numpy
- scikit-learn
- xgboost
- econml
- doubleml
- matplotlib

## Notes

File paths in the scripts should be updated before running the analyses.
