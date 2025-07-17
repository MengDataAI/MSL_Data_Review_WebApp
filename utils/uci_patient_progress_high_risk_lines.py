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

# Filter for HR (initial result >= 1% ) with atleast 2 tests results
# Filter mask
mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)

high_risk_df = sorted_first_year_df[mask_hr].copy()

# High risk table
# Create a row to count test number
high_risk_df['TEST_NUMBER'] = high_risk_df.groupby(['PID']).cumcount() +1

# Proportion of High RiskPatients with Subsequent Test >=1%
high_risk_subsequent_elevated = high_risk_df.loc[(high_risk_df['TEST_NUMBER'] != 1) & (high_risk_df['RES_RESULT_NUMERIC'] >= 1)]['PID'].nunique()

high_risk_unique_pid = high_risk_df['PID'].nunique()

high_risk_subsequent = round((high_risk_subsequent_elevated/high_risk_unique_pid)*100,1)

print(f'{high_risk_subsequent}% of high risk patients ({high_risk_subsequent_elevated}/{high_risk_unique_pid}) had a subsequent test ≥ 1%')


# Get list of Patients in High Risk Group that had a Subsequent Elevated AlloSure Result
pid_filter_high = high_risk_df.loc[(high_risk_df['TEST_NUMBER'] != 1) & (high_risk_df['RES_RESULT_NUMERIC'] >= 1), 'PID']; high_risk_df[['PID', 'RES_RESULT_NUMERIC', 'DRAW_DATE', 'RESULT_POST_TRANSPLANT', 'TEST_NUMBER']].loc[high_risk_df['PID'].isin(pid_filter_high)]

pid_filter_high = pd.DataFrame(pid_filter_high)

# Filter the High Risk Dataframe for these Patients
high_risk_elevated_df = high_risk_df.loc[high_risk_df['PID'].isin(pid_filter_high['PID'])].copy()


# Line Plot of the High Risk Patients - With Subsequent Elevated AlloSure Result
# Create a line plot
# Create shading intervals for months
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=high_risk_elevated_df, x='RESULT_POST_TRANSPLANT', y='RES_RESULT_NUMERIC', hue='PID', markers= True, markersize= 8,dashes=False, alpha=0.8, linewidth=0.8, zorder=2, marker= 'X')

# # Add a tick mark at y=1
ticks = plt.yticks()[0].tolist()  # Get current y-ticks
if 1 not in ticks:
    ticks.append(1)
    ticks = sorted(ticks)
plt.yticks(ticks)

plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)

intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]

# Apply alternating shading
for i, (start, end) in enumerate(intervals):
    color = 'grey' if i % 2 == 0 else 'white'
    ax.axvspan(start, end, color=color, alpha=0.35)

# Force x-axis range
ax.set_xlim(0, 395)

# Show legend 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Patient ID')

plt.title('High Risk Patients with Subsequent AlloSure Score ≥ 1%')
plt.xlabel("Days Post-Transplant")
plt.ylabel("AlloSure (% dd-cfDNA)")
plt.show()
# Note to Victor: Adjust the x-axis to label by Protocol Testing Month



