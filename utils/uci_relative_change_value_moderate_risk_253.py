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

sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
group_patients = sorted_first_year_df.groupby('PID')
count = group_patients['RES_RESULT_NUMERIC'].transform('size')
first_record = group_patients.transform('first')

# Create Moderate Risk df
# Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
# Filter mask
mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
moderate_risk_df = sorted_first_year_df[mask_mr].copy()

rcv_df_moderate = moderate_risk_df.copy()

rcv_df_moderate['TEST_NUMBER'] = rcv_df_moderate.groupby('PID').cumcount() + 1

# Calculate Relative Between Tests for Each Patient (ie test 2-1/1 and test 3-2/2)
rcv_df_moderate['RELATIVE_BETWEEN_TEST'] = round(rcv_df_moderate.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

# For Moderate Patient Group
# Create RCV dataframe - exclude NA (first result)
rcv_plot_moderate = rcv_df_moderate.loc[rcv_df_moderate['RELATIVE_BETWEEN_TEST'].notna()].copy()

# Calculate the percent of RELATIVE_BETWEEN_TEST >=253, >=149, >=61 AND <149, <61
rcv_plot_moderate['RCV'] = rcv_plot_moderate['RELATIVE_BETWEEN_TEST'].apply(lambda x: '≥253' if x >= 253 
                                                          else ('≥149' if x >= 149 
                                                                else ('≥61' if x >= 61 
                                                                      else '<61')))



# Create Dataframe for the Patients with change > 253%
high_change_patients_moderate = rcv_plot_moderate[rcv_plot_moderate['RELATIVE_BETWEEN_TEST'] > 253]['PID'].unique()

# Filter for Patients
high_change_patients_moderate_filtered_df = moderate_risk_df[moderate_risk_df['PID'].isin(high_change_patients_moderate)]

# Line Plot of the Moderate Risk Patients with change > 253%
# Create a line plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=high_change_patients_moderate_filtered_df, x='RESULT_POST_TRANSPLANT', y='RES_RESULT_NUMERIC', hue='PID', markers= True, markersize= 8,dashes=False, alpha=0.8, linewidth=0.8, zorder=2, marker= 'X')

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

# Show legend 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Patient ID')

plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1)

plt.title('Moderate Risk Patients with Sequential AlloSure Score RCV >253%')
plt.xlabel("Days Post-Transplant")
plt.ylabel("AlloSure (% dd-cfDNA)")
plt.show()