
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


# Create Moderate Risk df
# Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
# Filter mask
mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
moderate_risk_df = sorted_first_year_df[mask_mr].copy()

# Create Low Risk df
# Filter for LR (initial result < 0.5%) with atleast 2 tests results
# Filter mask
mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
low_risk_df = sorted_first_year_df[mask_lr].copy()


# Create stacked dataframe
low_risk_df['RISK_DF'] = 'Low Risk'
moderate_risk_df['RISK_DF'] = 'Moderate Risk'
high_risk_df['RISK_DF'] = 'High Risk'

stacked_df = pd.concat([low_risk_df, moderate_risk_df, high_risk_df], ignore_index=True)

# Filter for less than 16
stacked_df = stacked_df[stacked_df['RES_RESULT_NUMERIC'] < 16]

# Violin Plot - All Groups
fig, ax = plt.subplots(figsize=(12, 5))

sns.violinplot(
    data = stacked_df, x="PROTOCOL_TESTING_MONTH", y="RES_RESULT_NUMERIC", order=order, cut=0, zorder=2, hue='RISK_DF'
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

plt.legend(title=None)
ax.set_title('AlloSure Results Across Low/Moderate/High Risk Groups')
ax.set_xlabel("Protocol Testing Month")
ax.set_ylabel("AlloSure (% dd-cfDNA)")
fig.show()