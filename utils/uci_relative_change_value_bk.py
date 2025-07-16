import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyspark.sql import SparkSession

import plotly.graph_objects as go
import plotly.express as px

# For All Patients
# Add Test Number for each Patients
rcv_df = first_year_df.copy()

rcv_df['TEST_NUMBER'] = rcv_df.groupby('PID').cumcount() + 1

# Calculate Relative Between Tests for Each Patient (ie test 2-1/1 and test 3-2/2)
rcv_df['RELATIVE_BETWEEN_TEST'] = round(rcv_df.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

# rcv_df[['PID', 'RESULT_POST_TRANSPLANT', 'RES_RESULT_NUMERIC', 'RELATIVE_BETWEEN_TEST', 'TEST_NUMBER']].head(20)

# For All Patients
# Create RCV dataframe - exclude NA (first result)
rcv_plot = rcv_df.loc[rcv_df['RELATIVE_BETWEEN_TEST'].notna()].copy()

# Calculate the percent of RELATIVE_BETWEEN_TEST >=253, >=149, >=61 AND <149, <61
rcv_plot['RCV'] = rcv_plot['RELATIVE_BETWEEN_TEST'].apply(lambda x: '≥253' if x >= 253 
                                                          else ('≥149' if x >= 149 
                                                                else ('≥61' if x >= 61 
                                                                      else '<61')))

# Create Histogram of Relative Change Between Tests - All Patients
plt.figure(figsize=(10, 5))
sns.histplot(data=rcv_plot, x='RELATIVE_BETWEEN_TEST', bins=60, color='black', edgecolor='white', linewidth=.8, zorder=2, alpha=.8)

# Shade background but not plot
plt.axvspan(rcv_plot['RELATIVE_BETWEEN_TEST'].min(), 61, color='green', alpha=0.2)
plt.axvspan(61, 149, color='yellow', alpha=0.2)
plt.axvspan(149, rcv_plot['RELATIVE_BETWEEN_TEST'].max(), color='red', alpha=0.1)

plt.axvline(x=61, color='green', linestyle='--', linewidth=1)
plt.axvline(x=149, color='yellow', linestyle='--', linewidth=1)

# X axis end on min and max
plt.xlim(rcv_plot['RELATIVE_BETWEEN_TEST'].min(), rcv_plot['RELATIVE_BETWEEN_TEST'].max())

# Overlay rcv_groupby as a table on this plot
table = plt.table(
    cellText=rcv_summary.values,
    colLabels=rcv_summary.columns,
    bbox=[1.01, .6, 0.4, 0.4],
    cellLoc='center',
    loc='upper right'
)

table.auto_set_font_size(False)
table.set_fontsize(9) 

plt.title('Distribution of Relative Change in Between Tests')
plt.xlabel('Relative Change Between Sequential Tests (%)')
plt.ylabel('Frequency')
plt.show()










