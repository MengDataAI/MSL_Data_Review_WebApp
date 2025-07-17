

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

# Violin Plot - High Risk Patients
fig, ax = plt.subplots(figsize=(9, 5))

sns.violinplot(
    data = high_risk_df, x="PROTOCOL_TESTING_MONTH", y="RES_RESULT_NUMERIC", order=order, cut=0
)

# Add a horizontal line at y=1
plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)

# # Add a tick mark at y=1
ticks = plt.yticks()[0].tolist()  # Get current y-ticks
if 1 not in ticks:
    ticks.append(1)
    ticks = sorted(ticks)
plt.yticks(ticks)

ax.set_title('High Risk Patients with Initial AlloSure Score ≥ 1%')
ax.set_xlabel("Protocol Testing Month")
ax.set_ylabel("AlloSure (% dd-cfDNA)")
fig.show()