import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyspark.sql import SparkSession

import plotly.graph_objects as go
import plotly.express as px


# Read in spark df and convert into pandas
spark_df = spark.read.table("production.data_engineering.UCI_ANALYSIS_PIPELINE") 
df = spark_df.toPandas()

# Filter dataset on valid first year months 
first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()]

# Create High Risk df
# Global logic applied to all risk groups 
sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
group_patients = sorted_first_year_df.groupby('PID')
count = group_patients['RES_RESULT_NUMERIC'].transform('size')
first_record = group_patients.transform('first')

# Create Low Risk df
# Filter for LR (initial result < 0.5%) with atleast 2 tests results
# Filter mask
mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
low_risk_df = sorted_first_year_df[mask_lr].copy()

# Count Distinct HR Patients
print(f'Total Low Risk Patients: {low_risk_df["PID"].nunique()}')


# AlloSure Results Summary Statistis - LR
summary_stats_lr = low_risk_df.groupby('PROTOCOL_TESTING_MONTH').agg(
    N = ('ACCESSIONID', 'nunique'),
    Median = ("RES_RESULT_NUMERIC", "median"),
    Min = ("RES_RESULT_NUMERIC", "min"),
    Max = ("RES_RESULT_NUMERIC", "max"),
    p_25 = ("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.25)),
    p_75 = ("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.75))).reset_index()

summary_stats_lr['p_25'] = summary_stats_lr['p_25'].round(2)
summary_stats_lr['p_75'] = summary_stats_lr['p_75'].round(2)
summary_stats_lr['Min'] = summary_stats_lr['Min'].round(2)
summary_stats_lr['Max'] = summary_stats_lr['Max'].round(2)
summary_stats_lr['Median'] = summary_stats_lr['Median'].round(2)

summary_stats_lr['Median (IQR)'] = (summary_stats_lr['Median'].astype(str) +  " (" + summary_stats_lr['p_25'].astype(str) + ", " + summary_stats_lr['p_75'].astype(str) + ")")
summary_stats_lr['Range'] = (summary_stats_lr['Min'].astype(str) + ", " + summary_stats_lr['Max'].astype(str))

# Drop columns 
summary_stats_lr.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)

# Order protocol months 
#order = [ 'M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']

first_col_lr = summary_stats_lr.columns[0]
summary_stats_lr[first_col_lr] = pd.Categorical(summary_stats_lr[first_col_lr], categories=order, ordered=True)
summary_stats_lr.sort_values(first_col_lr, inplace=True, ignore_index=True)

# Rotate table
rotated_lr = (summary_stats_lr.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(columns={'index': 'Protocol Testing Month'})

rotated_lr.head()

# Violin Plot - Low Risk Patients
fig, ax = plt.subplots(figsize=(9, 5))

sns.violinplot(
    data = low_risk_df, x="PROTOCOL_TESTING_MONTH", y="RES_RESULT_NUMERIC", order=order, cut=0, zorder=2
)

# # Add a tick mark at y=1
ticks = plt.yticks()[0].tolist()  # Get current y-ticks

# Ensure 0.5 and 1 are included
for tick in [0.5, 1]:
    if tick not in ticks:
        ticks.append(tick)

ticks = sorted(set(ticks))  # Sort and remove duplicates just in case
plt.yticks(ticks)

# Horizal lines and shading
plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1)

ax.set_title('Low Risk Patients with Initial AlloSure Score <0.5%')
ax.set_xlabel("Protocol Testing Month")
ax.set_ylabel("AlloSure (% dd-cfDNA)")
fig.show()


# Low risk table
# Create a row to count test number
low_risk_df['TEST_NUMBER'] = low_risk_df.groupby(['PID']).cumcount() +1

# Proportion of Low Risk Patients with Subsequent Test >=1%
low_risk_subsequent_elevated = low_risk_df.loc[(low_risk_df['TEST_NUMBER'] != 1) & (low_risk_df['RES_RESULT_NUMERIC'] >= 1)]['PID'].nunique()

low_risk_unique_pid = low_risk_df['PID'].nunique()

low_risk_subsequent = round((low_risk_subsequent_elevated/low_risk_unique_pid)*100,1)

print(f'{low_risk_subsequent}% of low risk patients ({low_risk_subsequent_elevated}/{low_risk_unique_pid}) had a subsequent test ≥ 1%')


# Get list of Patients in Low Risk Group that had a Subsequent Elevated AlloSure Result
pid_filter_low = low_risk_df.loc[(low_risk_df['TEST_NUMBER'] != 1) & (low_risk_df['RES_RESULT_NUMERIC'] >= 1), 'PID']; low_risk_df[['PID', 'RES_RESULT_NUMERIC', 'DRAW_DATE', 'RESULT_POST_TRANSPLANT', 'TEST_NUMBER']].loc[low_risk_df['PID'].isin(pid_filter_low)]

pid_filter_low = pd.DataFrame(pid_filter_low)

# Filter the Low Risk Dataframe for these Patients
low_risk_elevated_df = low_risk_df.loc[low_risk_df['PID'].isin(pid_filter_low['PID'])].copy()


# Calculate Average Time to Initial Elevated Result - Low Risk
# First Test
first_test_low = low_risk_elevated_df.loc[low_risk_elevated_df['TEST_NUMBER'] == 1][['PID', 'RESULT_POST_TRANSPLANT']].rename(columns={'RESULT_POST_TRANSPLANT': 'START_DAY'})

# Get first test where AS result is >=1%
as_100_low = low_risk_elevated_df.loc[low_risk_elevated_df['RES_RESULT_NUMERIC'] >= 1].groupby('PID').first().reset_index()

# Merge 
result_low = pd.merge(as_100_low[['PID', 'RESULT_POST_TRANSPLANT']], first_test_low, on='PID')

# Calculate Difference
result_low['TIME_TO_ELEVATED'] = result_low['RESULT_POST_TRANSPLANT'] - result_low['START_DAY']

# Calculate Statistics
mean_time_to_elevated_low = result_low['TIME_TO_ELEVATED'].mean()
median_time_to_elevated_low = result_low['TIME_TO_ELEVATED'].median()
ste_time_to_elevated_low = result_low['TIME_TO_ELEVATED'].sem()

# Calc Range
range_min_low  = result_low['TIME_TO_ELEVATED'].quantile(0.25)
range_max_low = result_low['TIME_TO_ELEVATED'].quantile(0.75)

# print(f'Average Time to Elevated Result: {mean_time_to_elevated_low:.0f} days')
# print(f'Median: {median_time_to_elevated_low:.0f} days')
# print(f'STE: {ste_time_to_elevated_low:.0f} days')
# print(f'IQR: {range_min_low:.0f} - {range_max_low:.0f} days')

# Use this for Sanity Check with result_low df
# moderate_risk_elevated_df[['PID', 'RESULT_POST_TRANSPLANT', 'TEST_NUMBER', 'RES_RESULT_STRING', 'RES_RESULT_NUMERIC']]


# Line Plot of the Low Risk Patients
# Create a line plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=low_risk_elevated_df, x='RESULT_POST_TRANSPLANT', y='RES_RESULT_NUMERIC', hue='PID', markers= True, markersize= 8,dashes=False, alpha=0.8, linewidth=0.8, zorder=2, marker= 'X')

# Add a tick mark at y=1
ticks = plt.yticks()[0].tolist()  # Get current y-ticks
if 1 not in ticks:
    ticks.append(1)
    ticks = sorted(ticks)
plt.yticks(ticks)

# # Add a tick mark at y=1
ticks = plt.yticks()[0].tolist()  # Get current y-ticks
if 1 not in ticks:
    ticks.append(1)
    ticks = sorted(ticks)
plt.yticks(ticks)

intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]

# Apply alternating shading
for i, (start, end) in enumerate(intervals):
    color = 'grey' if i % 2 == 0 else 'white'
    ax.axvspan(start, end, color=color, alpha=0.35)

# Force x-axis range
ax.set_xlim(0, 395)

# Add Summary Statistics Text
ax.text(0.99, 0.98, f"Average Time to Elevated Result (STE): {mean_time_to_elevated_low:.0f} ({ste_time_to_elevated_low:.0f})  days", color='black', fontsize=9, transform=ax.transAxes, ha='right', va='top',zorder=2)
ax.text(0.99, 0.94, f"Median (IQR): {median_time_to_elevated_low:.0f} ({range_min_low:.0f} - {range_max_low:.0f}) days", color='black', fontsize=9, transform=ax.transAxes, ha='right', va='top',zorder=2)

# Show legend 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Patient ID')

plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1)

plt.title('Low Risk Patients with Subsequent AlloSure Score ≥ 1%')
plt.xlabel("Days Post-Transplant")
plt.ylabel("AlloSure (% dd-cfDNA)")
plt.show()







