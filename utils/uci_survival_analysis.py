import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyspark.sql import SparkSession

import plotly.graph_objects as go
import plotly.express as px

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from scipy.stats import ttest_ind, chi2_contingency


# Read in spark df and convert into pandas
spark_df = spark.read.table("production.data_engineering.UCI_ANALYSIS_PIPELINE") 
df = spark_df.toPandas()

# Filter dataset on valid first year months 
first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()]


df = first_year_df[['PID', 'ACCESSIONID', 'RESULT_POST_TRANSPLANT', 'DONOR_TYPE', 'RES_RESULT_NUMERIC', 'ON_PROTOCOL', 'PROTOCOL_TESTING_MONTH', 'TRANSPLANT_AGE', 'SEX']].copy()

# Convert Transplant Age to Int
df['TRANSPLANT_AGE'] = df['TRANSPLANT_AGE'].astype(int)


# Filter for only patients on protocol and for results in the first year. There is 130 Patients.
df = df[(df['ON_PROTOCOL'] == 'Y') & (df['PROTOCOL_TESTING_MONTH'].notna())]

# Order dataframe by PID and RESULT_POST_TRANSPLANT
df = df.sort_values(by=['PID', 'RESULT_POST_TRANSPLANT'])

# Add a Test Count Column
df['TEST_NUMBER'] = df.groupby(['PID']).cumcount() +1


# Global logic applied to all risk groups 
sorted_first_year_df = df.sort_values(['PID', 'TEST_NUMBER']).copy()
group_patients = sorted_first_year_df.groupby('PID')
count = group_patients['RES_RESULT_NUMERIC'].transform('size')
first_record = group_patients.transform('first')

# Moderate Risk
mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)

mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)

# Create a column in dataframe to tag mask_mr and mask_lr
df['RISK_GROUP'] = np.where(mask_mr, 'MR', np.where(mask_lr, 'LR', 'EXCLUDE'))

# Filter out Nan in Risk Group
df = df[df['RISK_GROUP'] != 'EXCLUDE']


# Create TIME_TO_EVENT Column as well as Binary classification for whether event happened (initial result >=1%, where 1 = event occured)
df['TIME_TO_EVENT'] = None
df['EVENT_OCCURRED'] = None

for pid, patient_data in df.groupby('PID'):

    patient_data = patient_data.sort_values(['PID','RESULT_POST_TRANSPLANT'])

    over_cutoff = patient_data[patient_data['RES_RESULT_NUMERIC'] >= 1]

    if not over_cutoff.empty:

        # Look for the first time patient receive AS score >=1%
        time_to_event = over_cutoff.iloc[0]['RESULT_POST_TRANSPLANT']
        event_occurred = 1

    else:
        # If not available then take max result post transplant (last test), and event occurred = 0
        time_to_event = patient_data['RESULT_POST_TRANSPLANT'].max()
        event_occurred = 0

    df.loc[(df['PID'] == pid), 'TIME_TO_EVENT'] = time_to_event
    df.loc[(df['PID'] == pid), 'EVENT_OCCURRED'] = event_occurred


# Convert to Int
df['TIME_TO_EVENT'] = df['TIME_TO_EVENT'].astype(int)
df['EVENT_OCCURRED'] = df['EVENT_OCCURRED'].astype(int)




# One-hot encode DONOR_TYPE and TIME_TO_FIRST_RESULT
df_encoded = pd.get_dummies(df, columns =['DONOR_TYPE', 'RISK_GROUP', 'SEX'], drop_first=False)

# Convert the one hot encoded columns to integer
df_encoded[['DONOR_TYPE_Deceased', 'DONOR_TYPE_Living','RISK_GROUP_MR', 'RISK_GROUP_LR', 'SEX_Male', 'SEX_Female']] = df_encoded[['DONOR_TYPE_Deceased', 'DONOR_TYPE_Living','RISK_GROUP_MR', 'RISK_GROUP_LR', 'SEX_Male', 'SEX_Female']].astype(int)

# Select columns for analysis and drop duplicates, reference is DONOR TYPE = Living, Low Risk Group, Sex Male
df_encoded = df_encoded[['PID', 'TIME_TO_EVENT', 'EVENT_OCCURRED', 'DONOR_TYPE_Deceased', 'RISK_GROUP_MR', 'SEX_Female', 'TRANSPLANT_AGE']]
df_encoded = df_encoded.drop_duplicates()

df_encoded = df_encoded[['TIME_TO_EVENT', 'EVENT_OCCURRED', 'DONOR_TYPE_Deceased', 'RISK_GROUP_MR', 'SEX_Female', 'TRANSPLANT_AGE']]


# Initialize and fit the model
cph = CoxPHFitter()
cph.fit(df_encoded, duration_col='TIME_TO_EVENT', event_col='EVENT_OCCURRED')

# Summary of the model
cph.print_summary()


ax = cph.plot(
  # adds confidence interval lines
    linewidth=5,
    c="red"
)

plt.title("Hazard Ratios with 95% Confidence Intervals", fontsize=16, fontweight='bold')
plt.xlabel("log(HR)", fontsize=14)

# Customize grid
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# Dataframe for Kaplan Meier Curves
km_df = df[['PID', 'TIME_TO_EVENT', 'EVENT_OCCURRED', 'RISK_GROUP', 'DONOR_TYPE', 'SEX']].copy()
km_df = km_df.drop_duplicates()
km_df = km_df.sort_values(by=['PID', 'TIME_TO_EVENT'])


# Plot KM Curve
# Split into two groups
group_MR = km_df[km_df['RISK_GROUP'] == 'MR']
group_LR = km_df[km_df['RISK_GROUP'] == 'LR']

fig = plt.figure(figsize=(10, 6))

# Initialize the KaplanMeierFitter
kmf_mr = KaplanMeierFitter()
kmf_lr = KaplanMeierFitter()

# Plot Moderate Risk
kmf_mr.fit(group_MR['TIME_TO_EVENT'], event_observed=group_MR['EVENT_OCCURRED'], label='Moderate Risk')
ax = kmf_mr.plot_survival_function(show_censors=True, censor_styles={'ms': 7, 'marker': 'x'})

# Plot Low Risk
kmf_lr.fit(group_LR['TIME_TO_EVENT'], event_observed=group_LR['EVENT_OCCURRED'], label='Low Risk')
kmf_lr.plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 7, 'marker': 'x'})

# Add P-Value from the Cox Regression
p_value = cph.summary['p'].iloc[1]
plt.text(0.92, 0.95, f'P-value: {p_value:.4f}', transform=plt.gca().transAxes, ha='center', va='center')

# Adjust x tick label by 30 days
plt.xticks(range(0, 395, 30), [str(i) for i in range(0, 395, 30)])

plt.legend(loc='lower left')
plt.title('Survival Curves Between Moderate and Low Risk Groups')
plt.xlabel('Time')
plt.ylabel('Free from AS>1%')
# plt.grid(True)

# Table
add_at_risk_counts(kmf_mr, kmf_lr, ax=ax)

plt.tight_layout()
plt.show()




# Plot KM Curve
# Split into two groups
group_DECEASED = km_df[km_df['DONOR_TYPE'] == 'Deceased']
group_LIVING = km_df[km_df['DONOR_TYPE'] == 'Living']

fig = plt.figure(figsize=(10, 6))

# Initialize the KaplanMeierFitter
kmf_DECEASED = KaplanMeierFitter()
kmf_LIVING = KaplanMeierFitter()

# Plot Deceased
kmf_DECEASED.fit(group_DECEASED['TIME_TO_EVENT'], event_observed=group_DECEASED['EVENT_OCCURRED'], label='Deceased')
ax = kmf_DECEASED.plot_survival_function(show_censors=True, censor_styles={'ms': 7, 'marker': 'x'})

# Plot Over 1 Living
kmf_LIVING.fit(group_LIVING['TIME_TO_EVENT'], event_observed=group_LIVING['EVENT_OCCURRED'], label='Living')
kmf_LIVING.plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 7, 'marker': 'x'})

# Add P-Value from the Cox Regression
p_value = cph.summary['p'].iloc[0]
plt.text(0.92, 0.95, f'P-value: {p_value:.4f}', transform=plt.gca().transAxes, ha='center', va='center')

# Adjust x tick label by 30 days
plt.xticks(range(0, 395, 30), [str(i) for i in range(0, 395, 30)])

plt.legend(loc='lower left')
plt.title('Survival Curves Between Deceased and Living Donor Groups')
plt.xlabel('Time')
plt.ylabel('Free from AS>1%')
# plt.grid(True)

# Table 
add_at_risk_counts(kmf_DECEASED, kmf_LIVING, ax=ax)

plt.tight_layout()
plt.show()


