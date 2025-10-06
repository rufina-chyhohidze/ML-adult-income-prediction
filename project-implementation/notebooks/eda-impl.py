import numpy as np
import pandas as pd

# setting columns
column_names = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]
#reading from train and test data
df_train_data = pd.read_csv('data/adult.data', names=column_names)
df_test_data = pd.read_csv('data/adult.test', skiprows=1, names=column_names)

print(df_train_data.head(5))
print(df_test_data.head(5))

print(df_train_data.tail(5))
print(df_test_data.tail(5))

#inconsistent data check for all columns
print(df_train_data.info())
print(df_test_data.info())

print(df_train_data.describe())
print(df_test_data.describe())
print("---------------------------------------------------------------------------------------------------------------")
# conclusion from analysing each column
"""
| **Column**         | **Train Summary**                         | **Test Summary**          | **Conclusion / Action**                                                   |
| ------------------ | ----------------------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| **age**            | int64, 17–90, mean ≈ 38.6                 | int64, 17–90, mean ≈ 38.8 | ✅ Consistent. One invalid value fixed. Keep as numeric.                   |
| **workclass**      | object, some `"?"`                        | object, some `"?"`        | ✅ Same categories. Treat `"?"` as missing.                                |
| **fnlwgt**         | int64, mean ≈ 1.89e5                      | int64, mean ≈ 1.89e5      | ✅ Similar distribution. Can safely drop (survey weight).                  |
| **education**      | object (16 levels)                        | object (16 levels)        | ✅ Same categories. Will likely drop later (duplicate of `education-num`). |
| **education-num**  | int64, 1–16                               | int64, 1–16               | ✅ Matches. Keep numeric form.                                             |
| **marital-status** | object                                    | object                    | ✅ Same values. Keep.                                                      |
| **occupation**     | object, some `"?"`                        | object, some `"?"`        | ⚠️ Missing values exist. Fill with “Unknown” or drop rows.                |
| **relationship**   | object                                    | object                    | ✅ Same categories. Keep.                                                  |
| **race**           | object                                    | object                    | ✅ Consistent. Keep.                                                       |
| **sex**            | object (“Male”, “Female”)                 | object                    | ✅ Same categories. Keep.                                                  |
| **capital-gain**   | int64, highly skewed (many 0s, max 99999) | int64, same               | ✅ Same range. Consider log transform later.                               |
| **capital-loss**   | int64, mostly 0s                          | int64, same               | ✅ Same range. Keep.                                                       |
| **hours-per-week** | int64, 1–99, mean ≈ 40                    | int64, same               | ✅ Consistent. Keep numeric.                                               |
| **native-country** | object, many rare categories, some `"?"`  | object, same              | ⚠️ Handle `"?"` and consider grouping rare categories.                    |
| **income**         | `<=50K`, `>50K`                           | `<=50K.`, `>50K.`         | ⚠️ Clean test labels (remove trailing `.`). Then consistent.              |

"""
for col in ["workclass", "occupation", "native-country"]:
    df_train_data[col] = df_train_data[col].replace("?", "Unknown")
    df_test_data[col] = df_test_data[col].replace("?", "Unknown")

"""
| Column             | Why impute with `"Unknown"` instead of deleting                                                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **workclass**      | Missing values mean we don’t know the employment type. It could relate to income (e.g., unknown workclass might signal unstable jobs). Keeping it as `"Unknown"` lets the model capture that pattern. |
| **occupation**     | Closely tied to income level. Deleting would lose important salary-related info. `"Unknown"` lets the model treat missing occupation as a distinct, learnable group.                                  |
| **native-country** | Most are “United-States”; missing ones are rare. Deletion has no benefit, and `"Unknown"` keeps rows while preserving global consistency.                                                             |
"""

#Checking that we fixed the missing vals issues with imputation
cols = ["workclass", "occupation", "native-country"]

print("Column          | '?' in Train/Test | 'Unknown' in Train | Status")
print("-" * 70)

for col in cols:
    q_train = df_train_data[col].isin(["?"]).sum()
    q_test = df_test_data[col].isin(["?"]).sum()
    unk_train = (df_train_data[col] == "Unknown").sum()
    print(f"{col:15} | {q_train}/{q_test:<15} | {unk_train:<17} | {' Clean' if q_train==q_test==0 else ' Check again'}")

print("---------------------------------------------------------------------------------------------------------------")

#checking for inconsistencies with categories in each column between test and train data set
cat_cols = df_train_data.select_dtypes(include="object").columns

def check_categorical_inconsistencies(cat_cols):
    for col in cat_cols:
        train_cats = set(df_train_data[col].unique())
        test_cats = set(df_test_data[col].unique())
        only_in_train = train_cats - test_cats
        only_in_test = test_cats - train_cats
        print(f"\n {col}")
        print("  → Only in train:", only_in_train if only_in_train else "None")
        print("  → Only in test :", only_in_test if only_in_test else "None")

check_categorical_inconsistencies(cat_cols)
print("---------------------------------------------------------------------------------------------------------------")
# fixing inconst of categories accros these sets
# - leaving the one single different category of native country in data set adult.data because::
# #Why:
# It’s a single rare category (only one record).
# Dropping or renaming it won’t affect the model.
# During model training, algorithms will simply treat it as an uncommon category.
# --- Fix leading/trailing spaces everywhere ---
for df in [df_train_data, df_test_data]:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

# --- Fix income labels (ensure consistent formatting) ---
df_train_data["income"] = df_train_data["income"].str.replace(".", "", regex=False)
df_test_data["income"] = df_test_data["income"].str.replace(".", "", regex=False)

cat_cols = ["native-country", "income"]

check_categorical_inconsistencies(cat_cols)
print("---------------------------------------------------------------------------------------------------------------")

#we drop fnlwgt, thats why:
"""
it’s not an individual feature; it’s survey metadata.
Doesn’t help predict income.
Can add noise or bias.
"""
df_train_data.drop(columns="fnlwgt", inplace=True)
df_test_data.drop(columns="fnlwgt", inplace=True)

print(df_train_data.columns)
print(df_test_data.columns)
print("---------------------------------------------------------------------------------------------------------------")

#drop education, because
"""
education-num is numeric (ordinal) — models handle it directly.
education is text — redundant and adds no new info.
Using both duplicates the same signal and can distort feature importance."""

edu_check = df_train_data[["education", "education-num"]].drop_duplicates().sort_values("education-num")
print(edu_check)

df_train_data.drop(columns="education", inplace=True)
df_test_data.drop(columns="education", inplace=True)

print(df_train_data.columns)
print(df_test_data.columns)
print("---------------------------------------------------------------------------------------------------------------")

#important to notice that some columns like sex or race have low number of different categories which can help our model learn more clear patterns and has a low risk for overfitting
# and columns with high categories like occupatioon  (15) and native country (42) can have an efect on our model make it more complex or not generalize well, especiially if categories have very few samples
cat_cols = df_train_data.select_dtypes(include="object").columns.drop("income")

cat_summary = (
    df_train_data[cat_cols]
    .nunique()
    .sort_values()
    .reset_index()
    .rename(columns={"index": "column", 0: "unique_categories"})
)

print(cat_summary)
print("---------------------------------------------------------------------------------------------------------------")

"""
- What this means:
Almost everyone has 0 for both, so these columns are very sparse but still meaningful.
Nonzero values represent people who had investments or losses — often linked to higher income.
- What to do:
Keep both columns.
Don’t delete zeros; they carry meaning (“no capital gain/loss”).
Optionally apply log transform (np.log1p()) later to reduce skew if you use linear models (not needed for tree-based ones).
"""
for col in ["capital-gain", "capital-loss"]:
    unique_vals = df_train_data[col].nunique()
    zero_count = (df_train_data[col] == 0).sum()
    nonzero_unique = df_train_data.loc[df_train_data[col] != 0, col].nunique()
    print(f"{col}: {unique_vals} unique values ({zero_count} zeros, {nonzero_unique} nonzero unique values)")

print("---------------------------------------------------------------------------------------------------------------")
print(df_train_data["hours-per-week"].dtype)
print(df_train_data["hours-per-week"].describe())

#check missing values
print(df_train_data.isna().sum())

print("---------------------------------------------------------------------------------------------------------------")
import matplotlib.pyplot as plt

# --- 1️ Numeric Histograms ---
num_cols = ["age", "hours-per-week", "capital-gain", "capital-loss"]

plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    plt.hist(df_train_data[col], bins=30, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {col}", fontsize=12, fontweight="bold")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# --- 2️ Boxplots for numeric columns ---
plt.figure(figsize=(10,5))
df_train_data[num_cols].boxplot()
plt.title("Boxplots of Numeric Features", fontsize=14, fontweight="bold")
plt.ylabel("Value")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


# --- 3️ Target Variable Distribution ---
plt.figure(figsize=(5,4))
df_train_data["income"].value_counts().sort_index().plot(
    kind="bar", color=["lightcoral","lightgreen"], edgecolor="black"
)
plt.title("Income Class Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Income Category")
plt.ylabel("Number of Individuals")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
print("---------------------------------------------------------------------------------------------------------------")
#outliers

#%% md
### Determine whether there are outliers using the IQR method
#We’ll use the IQR method for **age**, **hours-per-week**, **capital-gain**, and **capital-loss** to detect potential outliers and interpret whether they are errors or valid data points.
#%%

for col in ["age", "hours-per-week", "capital-gain", "capital-loss"]:
    Q1 = df_train_data[col].quantile(0.25)
    Q3 = df_train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_train_data[(df_train_data[col] < lower_bound) | (df_train_data[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected (out of {len(df_train_data)})")

#%% md
### Interpretation
"""- **Age:** A few older individuals (>80) may appear as outliers, but they are real people — keep them.  
- **Hours-per-week:** Some work extreme hours (90–99). Valid but rare; keep them as real behavior examples.  
- **Capital-gain / Capital-loss:** Many zeros with a few very large values. These are legitimate financial values, not errors.  
Thus, we **keep** all outliers for now. We may try log-transforming these columns later to reduce skew."""
#%%

# Apply log1p transform for skewed financial features (optional for linear models)
df_train_data["capital-gain_log"] = np.log1p(df_train_data["capital-gain"])
df_train_data["capital-loss_log"] = np.log1p(df_train_data["capital-loss"])

for col in ["capital-gain_log", "capital-loss_log"]:
    Q1 = df_train_data[col].quantile(0.25)
    Q3 = df_train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_train_data[(df_train_data[col] < lower_bound) | (df_train_data[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected after log transform")

#%% md
"""After applying the log transformation, the **capital-gain** and **capital-loss** distributions become less skewed,
and the number of detected outliers decreases.  
This confirms that the extreme financial values are genuine but highly skewed, so the log transform helps stabilize them
for models that assume more normally distributed inputs."""

#%% md
# ### Visualizing Outliers Before and After Log Transformation
# We’ll use boxplots to see how outliers appear in **age**, **hours-per-week**, **capital-gain**, and **capital-loss**,
# and check how the log transformation affects the last two financial features.
#%%
import matplotlib.pyplot as plt

# --- Before transformation ---
plt.figure(figsize=(10,6))
df_train_data[["age","hours-per-week","capital-gain","capital-loss"]].boxplot()
plt.title("Boxplots of Numeric Features (Before Log Transform)", fontsize=13, fontweight="bold")
plt.ylabel("Value")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# --- After transformation ---
plt.figure(figsize=(8,5))
df_train_data[["capital-gain_log","capital-loss_log"]].boxplot()
plt.title("Boxplots of Financial Features (After Log Transform)", fontsize=13, fontweight="bold")
plt.ylabel("Log-scaled Value")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

"""
| Feature                                 | Outliers Found                | What the Plot Shows                                      | Decision                                                                                     |
| --------------------------------------- | ----------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **age**                                 | 143                           | A few older individuals (80–90).                         | ✅ Keep — real people, valid ages.                                                            |
| **hours-per-week**                      | 9008                          | Many >50 hours — long-hour workers.                      | ✅ Keep — not wrong, just variation. Maybe consider capping at 99 if extreme, but fine as-is. |
| **capital-gain**                        | 2712                          | Heavy right skew, a few very high values.                | ✅ Keep values, apply **log1p** transform to reduce skew.                                     |
| **capital-loss**                        | 1519                          | Similar pattern to gain, right skew.                     | ✅ Keep values, **log1p** transform improves spread.                                          |
| **capital-gain_log / capital-loss_log** | Same count but better scaling | Boxplots show compressed spread, fewer visible extremes. | ✅ Use **log-transformed** versions in models needing normality.                              |
"""








