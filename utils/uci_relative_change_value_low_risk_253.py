
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyspark.sql import SparkSession

import plotly.graph_objects as go
import plotly.express as px
# 
# Read in spark df and convert into pandas
spark_df = spark.read.table("production.data_engineering.UCI_ANALYSIS_PIPELINE") 
df = spark_df.toPandas()
# Filter dataset on valid first year months 
first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()]
# For All Patients
# Add Test Number for each Patients
rcv_df = first_year_df.copy()

rcv_df['TEST_NUMBER'] = rcv_df.groupby('PID').cumcount() + 1

# Calculate Relative Between Tests for Each Patient (ie test 2-1/1 and test 3-2/2)
rcv_df['RELATIVE_BETWEEN_TEST'] = round(rcv_df.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

# rcv_df[['PID', 'RESUL
# Sort by Patient ID and days post-transplant
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
# Modify for the Low Risk Group
rcv_df_low = low_risk_df.copy()

rcv_df_low['TEST_NUMBER'] = rcv_df_low.groupby('PID').cumcount() + 1

# Calculate Relative Between Tests for Each Patient (ie test 2-1/1 and test 3-2/2)
rcv_df_low['RELATIVE_BETWEEN_TEST'] = round(rcv_df_low.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

# rcv_df_low[['PID', 'RESULT_POST_TRANSPLANT', 'RES_RESULT_NUMERIC', 'RELATIVE_BETWEEN_TEST', 'TEST_NUMBER']].head(20)

# For Low Risk Group
# Create RCV dataframe - exclude NA (first result)
rcv_plot_low = rcv_df_low.loc[rcv_df_low['RELATIVE_BETWEEN_TEST'].notna()].copy()

# Calculate the percent of RELATIVE_BETWEEN_TEST >=253, >=149, >=61 AND <149, <61
rcv_plot_low['RCV'] = rcv_plot_low['RELATIVE_BETWEEN_TEST'].apply(lambda x: '≥253' if x >= 253 
                                                          else ('≥149' if x >= 149 
                                                                else ('≥61' if x >= 61 
                                                                      else '<61')))

# Group By RCV and get percent total
rcv_groupby_low = round(rcv_plot_low.groupby('RCV')['PID'].count()/len(rcv_plot_low)* 100,0)
rcv_group_sample_size_low = rcv_plot_low.groupby('RCV')['PID'].count()

# Order this table by <61, >=61, >=149, >=253
rcv_groupby_low = rcv_groupby_low.reindex(['<61', '≥61', '≥149', '≥253'])
rcv_group_sample_size_low = rcv_group_sample_size_low.reindex(['<61', '≥61', '≥149', '≥253'])

# Change to string 
rcv_groupby_low = rcv_groupby_low.astype(int).astype(str) + '%'
rcv_group_sample_size_low = rcv_group_sample_size_low.fillna(0).astype(int)

# Combined Table
rcv_summary_low = pd.DataFrame({
    '(N)': rcv_group_sample_size_low,
    'RCV': rcv_groupby_low
})

rcv_summary_low.index.name = "RCV Range (%)"
rcv_summary_low = rcv_summary_low.reset_index()

# Create Dataframe for the Patients with change > 253%
high_change_patients_low = rcv_plot_low[rcv_plot_low['RELATIVE_BETWEEN_TEST'] > 253]['PID'].unique()

# Filter for Patients
high_change_patients_low_filtered_df = low_risk_df[low_risk_df['PID'].isin(high_change_patients_low)]


# Line Plot of the Low Risk Patients with change > 253%
# Create a line plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=high_change_patients_low_filtered_df, x='RESULT_POST_TRANSPLANT', y='RES_RESULT_NUMERIC', hue='PID', markers= True, markersize= 8,dashes=False, alpha=0.8, linewidth=0.8, zorder=2, marker= 'X')

# Add a tick mark at y=1
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

# Show legend 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Patient ID')

plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1)

plt.title('Low Risk Patients with Sequential AlloSure Score RCV >253%')
plt.xlabel("Days Post-Transplant")
plt.ylabel("AlloSure (% dd-cfDNA)")
plt.show()