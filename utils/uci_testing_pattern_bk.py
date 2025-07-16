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

##########################################################
## Firgure 1. Histogram for Patients 
# Create Histogram for Patients on Protocol
fig, ax = plt.subplots(figsize=(14, 5))

plt.hist(first_year_df['RESULT_POST_TRANSPLANT'], color='darkblue', alpha=.8, bins=50, edgecolor='white', zorder=2)

# Change x-axis to have minor tick at ever 7 days
tick_positions = range(0, 400, 14)
tick_labels = [str(t) for t in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Create shading intervals for months
intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]

# Apply alternating shading
for i, (start, end) in enumerate(intervals):
    color = 'grey' if i % 2 == 0 else 'white'
    ax.axvspan(start, end, color=color, alpha=0.35)

# Force x-axis range
ax.set_xlim(0, 395)

plt.xlabel('Days Post-Transplant')
plt.ylabel('Frequency')
plt.title('AlloSure Testing Frequency')

############################################################
## Firgure 2. Number of Tests by First Year Schedule

# Count Unique Accession ID by Testing 
grouped_testing_month = first_year_df.groupby('PROTOCOL_TESTING_MONTH').agg({'ACCESSIONID': 'nunique', 'RESULT_POST_TRANSPLANT': 'mean'}).reset_index().sort_values(by='RESULT_POST_TRANSPLANT')

# Bar plot - Number of Tests
sns.color_palette()
fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    data = grouped_testing_month, x="PROTOCOL_TESTING_MONTH", y="ACCESSIONID", edgecolor='black'
)

# Bar label
for i, v in enumerate(grouped_testing_month['ACCESSIONID']):
    ax.text(i, v - 7, str(v), ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')

ax.set_title('Number of Tests by First Year Schedule')
ax.set_xlabel("Protocol Testing Month")
ax.set_ylabel("Number of Tests")
fig.show()


############################################################
## Firgure 3. Distribution of AlloSure (% dd-cfDNA) by First Year Schedule

# Violin Plot - AlloSure Results
fig, ax = plt.subplots(figsize=(9, 5))

sns.violinplot(
    data = first_year_df, x="PROTOCOL_TESTING_MONTH", y="RES_RESULT_NUMERIC", order=order, cut=0
)

# Add a horizontal line at y=1
plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)

# # Add a tick mark at y=1
ticks = plt.yticks()[0].tolist()  # Get current y-ticks
if 1 not in ticks:
    ticks.append(1)
    ticks = sorted(ticks)
plt.yticks(ticks)

ax.set_title('Distribution of AlloSure (% dd-cfDNA) by First Year Schedule')
ax.set_xlabel("Protocol Testing Month")
ax.set_ylabel("AlloSure (% dd-cfDNA)")
fig.show()



############################################################
## Statistics Table. Distribution of AlloSure (% dd-cfDNA) by First Year Schedule

# AlloSure Results Summary Statistis
summary_stats_2b = first_year_df.groupby('PROTOCOL_TESTING_MONTH').agg(
    N = ('ACCESSIONID', 'nunique'),
    Median = ("RES_RESULT_NUMERIC", "median"),
    Min = ("RES_RESULT_NUMERIC", "min"),
    Max = ("RES_RESULT_NUMERIC", "max"),
    p_25 = ("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.25)),
    p_75 = ("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.75))).reset_index()

summary_stats_2b['p_25'] = summary_stats_2b['p_25'].round(2)
summary_stats_2b['p_75'] = summary_stats_2b['p_75'].round(2)
summary_stats_2b['Min'] = summary_stats_2b['Min'].round(2)
summary_stats_2b['Max'] = summary_stats_2b['Max'].round(2)
summary_stats_2b['Median'] = summary_stats_2b['Median'].round(2)

summary_stats_2b['Median (IQR)'] = (summary_stats_2b['Median'].astype(str) +  " (" + summary_stats_2b['p_25'].astype(str) + ", " + summary_stats_2b['p_75'].astype(str) + ")")
summary_stats_2b['Range'] = ( summary_stats_2b['Min'].astype(str) + ", " + summary_stats_2b['Max'].astype(str))


# Drop columns 
summary_stats_2b.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)

# Order protocol months 
order = [ 'M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']

first_col = summary_stats_2b.columns[0]
summary_stats_2b[first_col] = pd.Categorical(summary_stats_2b[first_col], categories=order, ordered=True)
summary_stats_2b.sort_values(first_col, inplace=True, ignore_index=True)

# Rotate table
rotated = (summary_stats_2b.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(columns={'index': 'Protocol Testing Month'})

rotated.head()