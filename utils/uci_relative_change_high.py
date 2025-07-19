
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
from lifelines.statistics import logrank_test

from scipy.stats import ttest_ind, chi2_contingency

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

# Filter for HR (initial result >= 1% ) with atleast 2 tests results
# Filter mask
mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)

high_risk_df = sorted_first_year_df[mask_hr].copy()

rcv_df_high = high_risk_df.copy()

rcv_df_high['TEST_NUMBER'] = rcv_df_high.groupby('PID').cumcount() + 1

# Calculate Relative Between Tests for Each Patient (ie test 2-1/1 and test 3-2/2)
rcv_df_high['RELATIVE_BETWEEN_TEST'] = round(rcv_df_high.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

rcv_plot_high = rcv_df_high.loc[rcv_df_high['RELATIVE_BETWEEN_TEST'].notna()].copy()



# Create Dataframe for the Patients with change > 253%
high_change_patients_high = rcv_plot_high[rcv_plot_high['RELATIVE_BETWEEN_TEST'] > 253]['PID'].unique()

# Filter for Patients
high_change_patients_high_filtered_df = high_risk_df[high_risk_df['PID'].isin(high_change_patients_high)]

# Line Plot of the High Risk Patients with change > 253%
# Create a line plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=high_change_patients_high_filtered_df, x='RESULT_POST_TRANSPLANT', y='RES_RESULT_NUMERIC', hue='PID', markers= True, markersize= 8,dashes=False, alpha=0.8, linewidth=0.8, zorder=2, marker= 'X')

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

plt.title('High Risk Patients with Sequential AlloSure Score RCV >253%')
plt.xlabel("Days Post-Transplant")
plt.ylabel("AlloSure (% dd-cfDNA)")
plt.show()

