 

top20 = importance_df.head(20).copy()

# reverse for plotting
top20 = top20.iloc[::-1]
 

fig, ax = plt.subplots(figsize=(14, 10))

bars = ax.barh(
    top20["Feature"],
    top20["Importance"]
)

 

max_importance = top20["Importance"].max()

# add extra space on right side
ax.set_xlim(0, max_importance * 1.25)

 
for bar in bars:

    width = bar.get_width()

    ax.text(

        width + (max_importance * 0.015),

        bar.get_y() + bar.get_height()/2,

        f"{width:.4f}",

        va='center',

        fontsize=9

    )

 

ax.set_xlabel(
    "Random Forest Feature Importance",
    fontsize=12
)

ax.set_ylabel(
    "Features",
    fontsize=12
)

ax.set_title(
    f"Top 20 Random Forest Feature Importances for {OUTCOME.upper()}",
    fontsize=14,
    fontweight='bold'
)

 
plt.figtext(

    0.99,
    0.01,

    f"Best Cross-Validated R² = {best_r2:.4f}",

    horizontalalignment='right',
    fontsize=10

)

 

plt.tight_layout()

 

plt.savefig(
    f"RF_{OUTCOME.upper()}_Top20_Features.png",
    dpi=300,
    bbox_inches='tight'
)

 

plt.show()
 
 

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

 

df = pd.read_csv("combined_peaa.csv")

 

OUTCOME = "peaa"

y = df[OUTCOME]
 

X = df.drop(columns=[OUTCOME])
 

remove_keywords = [

    "cvl",
    "vload2",
    "esthivdate",
    "lg10cvl",
    "cvlfrhaart"

]

X = X.loc[:, ~X.columns.str.contains(
    "|".join(remove_keywords),
    case=False,
    regex=True
)]

 

print("Remaining Feature Matrix Shape:")
print(X.shape)

 

non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns

print("\nRemoved Non-Numeric Columns:")
print(non_numeric.tolist())

X = X.select_dtypes(include=[np.number])

 

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

 

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

 

param_dist = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 20, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2"]

}

 

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_dist,

    n_iter=30,

    cv=5,

    scoring="r2",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

 

random_search.fit(X_imputed, y)

 

best_rf = random_search.best_estimator_

best_r2 = random_search.best_score_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validated R²:")
print(best_r2)

 

importance_df = pd.DataFrame({

    "Feature": X_imputed.columns,

    "Importance": best_rf.feature_importances_

})

 

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

 

print("\nTop 30 Important Features:\n")

print(importance_df.head(30))

 

importance_df.to_csv(
    f"RF_{OUTCOME.upper()}_Feature_Importance.csv",
    index=False
)

 

top20 = importance_df.head(20).copy()

# reverse for plotting
top20 = top20.iloc[::-1]

 

fig, ax = plt.subplots(figsize=(14, 10))

bars = ax.barh(
    top20["Feature"],
    top20["Importance"]
)

 

max_importance = top20["Importance"].max()

ax.set_xlim(0, max_importance * 1.25)

 

for bar in bars:

    width = bar.get_width()

    ax.text(

        width + (max_importance * 0.015),

        bar.get_y() + bar.get_height()/2,

        f"{width:.4f}",

        va='center',

        fontsize=9

    )

 

ax.set_xlabel(
    "Random Forest Feature Importance",
    fontsize=12
)

ax.set_ylabel(
    "Features",
    fontsize=12
)

ax.set_title(
    f"Top 20 Random Forest Feature Importances for {OUTCOME.upper()}",
    fontsize=14,
    fontweight='bold'
)

 

plt.figtext(

    0.99,
    0.01,

    f"Best Cross-Validated R² = {best_r2:.4f}",

    horizontalalignment='right',
    fontsize=10

)

 

plt.tight_layout()

 

plt.savefig(
    f"RF_{OUTCOME.upper()}_Top20_Features.png",
    dpi=300,
    bbox_inches='tight'
)

 

plt.show()

 

threshold = importance_df["Importance"].median()

selected_features = importance_df[
    importance_df["Importance"] > threshold
]

print("\nSelected Features Above Median Importance:\n")

print(selected_features)

 

selected_features.to_csv(
    f"Selected_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)

 

top20.to_csv(
    f"Top20_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)

 

print("\nAnalysis Complete.")

print("\nGenerated Files:")

print(f"- RF_{OUTCOME.upper()}_Feature_Importance.csv")

print(f"- RF_{OUTCOME.upper()}_Top20_Features.png")

print(f"- Selected_RF_{OUTCOME.upper()}_Features.csv")

print(f"- Top20_RF_{OUTCOME.upper()}_Features.csv")
 

 

df = pd.read_csv("combined_eeaa.csv")
 

OUTCOME = "eeaa"

y = df[OUTCOME]

 

X = df.drop(columns=[OUTCOME])

 

remove_keywords = [

    "cvl",
    "vload2",
    "esthivdate",
    "lg10cvl",
    "cvlfrhaart"

]

X = X.loc[:, ~X.columns.str.contains(
    "|".join(remove_keywords),
    case=False,
    regex=True
)]

 

print("Remaining Feature Matrix Shape:")
print(X.shape)
 

non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns

print("\nRemoved Non-Numeric Columns:")
print(non_numeric.tolist())

X = X.select_dtypes(include=[np.number])

 

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

 

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)
 

param_dist = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 20, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2"]

}
 

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_dist,

    n_iter=30,

    cv=5,

    scoring="r2",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

 

random_search.fit(X_imputed, y)

 
best_rf = random_search.best_estimator_

best_r2 = random_search.best_score_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validated R²:")
print(best_r2)

 

importance_df = pd.DataFrame({

    "Feature": X_imputed.columns,

    "Importance": best_rf.feature_importances_

})

 

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

 

print("\nTop 30 Important Features:\n")

print(importance_df.head(30))
 

importance_df.to_csv(
    f"RF_{OUTCOME.upper()}_Feature_Importance.csv",
    index=False
)

 

top20 = importance_df.head(20).copy()

# reverse for plotting
top20 = top20.iloc[::-1]
 

fig, ax = plt.subplots(figsize=(14, 10))

bars = ax.barh(
    top20["Feature"],
    top20["Importance"]
)

 

max_importance = top20["Importance"].max()

ax.set_xlim(0, max_importance * 1.25)

 

for bar in bars:

    width = bar.get_width()

    ax.text(

        width + (max_importance * 0.015),

        bar.get_y() + bar.get_height()/2,

        f"{width:.4f}",

        va='center',

        fontsize=9

    )

 

ax.set_xlabel(
    "Random Forest Feature Importance",
    fontsize=12
)

ax.set_ylabel(
    "Features",
    fontsize=12
)

ax.set_title(
    f"Top 20 Random Forest Feature Importances for {OUTCOME.upper()}",
    fontsize=14,
    fontweight='bold'
)
 

plt.figtext(

    0.99,
    0.01,

    f"Best Cross-Validated R² = {best_r2:.4f}",

    horizontalalignment='right',
    fontsize=10

)

 

plt.tight_layout()

 
plt.savefig(
    f"RF_{OUTCOME.upper()}_Top20_Features.png",
    dpi=300,
    bbox_inches='tight'
)

 

plt.show()
 

threshold = importance_df["Importance"].median()

selected_features = importance_df[
    importance_df["Importance"] > threshold
]

print("\nSelected Features Above Median Importance:\n")

print(selected_features)

 

selected_features.to_csv(
    f"Selected_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)

 

top20.to_csv(
    f"Top20_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)

 
print("\nAnalysis Complete.")

print("\nGenerated Files:")

print(f"- RF_{OUTCOME.upper()}_Feature_Importance.csv")

print(f"- RF_{OUTCOME.upper()}_Top20_Features.png")

print(f"- Selected_RF_{OUTCOME.upper()}_Features.csv")

print(f"- Top20_RF_{OUTCOME.upper()}_Features.csv")
 

df = pd.read_csv("combined_aar.csv")

 

OUTCOME = "aar"

y = df[OUTCOME]

 

X = df.drop(columns=[OUTCOME])

 

remove_keywords = [

    "cvl",
    "vload2",
    "esthivdate",
    "lg10cvl",
    "cvlfrhaart"

]

X = X.loc[:, ~X.columns.str.contains(
    "|".join(remove_keywords),
    case=False,
    regex=True
)]

 

print("Remaining Feature Matrix Shape:")
print(X.shape)

 

non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns

print("\nRemoved Non-Numeric Columns:")
print(non_numeric.tolist())

X = X.select_dtypes(include=[np.number])

 

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

 

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

 
param_dist = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 20, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2"]

}

 

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_dist,

    n_iter=30,

    cv=5,

    scoring="r2",

    verbose=2,

    random_state=42,

    n_jobs=-1

)
 

random_search.fit(X_imputed, y)

 

best_rf = random_search.best_estimator_

best_r2 = random_search.best_score_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validated R²:")
print(best_r2)
 

importance_df = pd.DataFrame({

    "Feature": X_imputed.columns,

    "Importance": best_rf.feature_importances_

})

 
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
 
print("\nTop 30 Important Features:\n")

print(importance_df.head(30))

 

importance_df.to_csv(
    f"RF_{OUTCOME.upper()}_Feature_Importance.csv",
    index=False
)

 

top20 = importance_df.head(20).copy()

# reverse for plotting
top20 = top20.iloc[::-1]

 
fig, ax = plt.subplots(figsize=(14, 10))

bars = ax.barh(
    top20["Feature"],
    top20["Importance"]
)



max_importance = top20["Importance"].max()

ax.set_xlim(0, max_importance * 1.25)



for bar in bars:

    width = bar.get_width()

    ax.text(

        width + (max_importance * 0.015),

        bar.get_y() + bar.get_height()/2,

        f"{width:.4f}",

        va='center',

        fontsize=9

    )



ax.set_xlabel(
    "Random Forest Feature Importance",
    fontsize=12
)

ax.set_ylabel(
    "Features",
    fontsize=12
)

ax.set_title(
    f"Top 20 Random Forest Feature Importances for {OUTCOME.upper()}",
    fontsize=14,
    fontweight='bold'
)



plt.figtext(

    0.99,
    0.01,

    f"Best Cross-Validated R² = {best_r2:.4f}",

    horizontalalignment='right',
    fontsize=10

)



plt.tight_layout()



plt.savefig(
    f"RF_{OUTCOME.upper()}_Top20_Features.png",
    dpi=300,
    bbox_inches='tight'
)


plt.show()



threshold = importance_df["Importance"].median()

selected_features = importance_df[
    importance_df["Importance"] > threshold
]

print("\nSelected Features Above Median Importance:\n")

print(selected_features)



selected_features.to_csv(
    f"Selected_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)



top20.to_csv(
    f"Top20_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)



print("\nAnalysis Complete.")

print("\nGenerated Files:")

print(f"- RF_{OUTCOME.upper()}_Feature_Importance.csv")

print(f"- RF_{OUTCOME.upper()}_Top20_Features.png")

print(f"- Selected_RF_{OUTCOME.upper()}_Features.csv")

print(f"- Top20_RF_{OUTCOME.upper()}_Features.csv")


df = pd.read_csv("combined_dnamtladjage.csv")



OUTCOME = "dnamtladjage"

y = df[OUTCOME]


X = df.drop(columns=[OUTCOME])



remove_keywords = [

    "cvl",
    "vload2",
    "esthivdate",
    "lg10cvl",
    "cvlfrhaart"

]

X = X.loc[:, ~X.columns.str.contains(
    "|".join(remove_keywords),
    case=False,
    regex=True
)]



print("Remaining Feature Matrix Shape:")
print(X.shape)



non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns

print("\nRemoved Non-Numeric Columns:")
print(non_numeric.tolist())

X = X.select_dtypes(include=[np.number])



imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)



param_dist = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 20, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2"]

}



random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_dist,

    n_iter=30,

    cv=5,

    scoring="r2",

    verbose=2,

    random_state=42,

    n_jobs=-1

)



random_search.fit(X_imputed, y)



best_rf = random_search.best_estimator_

best_r2 = random_search.best_score_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validated R²:")
print(best_r2)


importance_df = pd.DataFrame({

    "Feature": X_imputed.columns,

    "Importance": best_rf.feature_importances_

})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)



print("\nTop 30 Important Features:\n")

print(importance_df.head(30))



importance_df.to_csv(
    f"RF_{OUTCOME.upper()}_Feature_Importance.csv",
    index=False
)



top20 = importance_df.head(20).copy()

# reverse for plotting
top20 = top20.iloc[::-1]



fig, ax = plt.subplots(figsize=(14, 10))

bars = ax.barh(
    top20["Feature"],
    top20["Importance"]
)



max_importance = top20["Importance"].max()

ax.set_xlim(0, max_importance * 1.25)


for bar in bars:

    width = bar.get_width()

    ax.text(

        width + (max_importance * 0.015),

        bar.get_y() + bar.get_height()/2,

        f"{width:.4f}",

        va='center',

        fontsize=9

    )



ax.set_xlabel(
    "Random Forest Feature Importance",
    fontsize=12
)

ax.set_ylabel(
    "Features",
    fontsize=12
)

ax.set_title(
    f"Top 20 Random Forest Feature Importances for {OUTCOME}",
    fontsize=14,
    fontweight='bold'
)



plt.figtext(

    0.99,
    0.01,

    f"Best Cross-Validated R² = {best_r2:.4f}",

    horizontalalignment='right',
    fontsize=10

)



plt.tight_layout()



plt.savefig(
    f"RF_{OUTCOME.upper()}_Top20_Features.png",
    dpi=300,
    bbox_inches='tight'
)



plt.show()


threshold = importance_df["Importance"].median()

selected_features = importance_df[
    importance_df["Importance"] > threshold
]

print("\nSelected Features Above Median Importance:\n")

print(selected_features)



selected_features.to_csv(
    f"Selected_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)



top20.to_csv(
    f"Top20_RF_{OUTCOME.upper()}_Features.csv",
    index=False
)



print("\nAnalysis Complete.")

print("\nGenerated Files:")

print(f"- RF_{OUTCOME.upper()}_Feature_Importance.csv")

print(f"- RF_{OUTCOME.upper()}_Top20_Features.png")

print(f"- Selected_RF_{OUTCOME.upper()}_Features.csv")

print(f"- Top20_RF_{OUTCOME.upper()}_Features.csv")
