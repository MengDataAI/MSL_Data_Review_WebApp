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

# Read in spark df and convert into pandas
spark_df = spark.read.table("production.data_engineering.UCI_ANALYSIS_PIPELINE") 
df = spark_df.toPandas()

# Create a copy of the first_record_df
sankey_df = first_year_df.copy()

# Add Test Number for each Patients
sankey_df['TEST_NUMBER'] = sankey_df.groupby('PID').cumcount() + 1

# Create field to assign unique value which we can use for defining the source and target variables
sankey_df['SANKEY'] = sankey_df.apply(lambda x: 0 if x['PROTOCOL_TESTING_MONTH'] == 'M1' and x['RISK_LEVEL'] == 'HR' 
                                      else 1 if x['PROTOCOL_TESTING_MONTH'] == 'M1' and x['RISK_LEVEL'] == 'MR' 
                                      else 2 if x['PROTOCOL_TESTING_MONTH'] == 'M1' and x['RISK_LEVEL'] == 'LR'
                                      else 3 if x['PROTOCOL_TESTING_MONTH'] == 'M2' and x['RISK_LEVEL'] == 'HR' 
                                      else 4 if x['PROTOCOL_TESTING_MONTH'] == 'M2' and x['RISK_LEVEL'] == 'MR' 
                                      else 5 if x['PROTOCOL_TESTING_MONTH'] == 'M2' and x['RISK_LEVEL'] == 'LR'
                                      else 6 if x['PROTOCOL_TESTING_MONTH'] == 'M3' and x['RISK_LEVEL'] == 'HR'
                                      else 7 if x['PROTOCOL_TESTING_MONTH'] == 'M3' and x['RISK_LEVEL'] == 'MR'
                                      else 8 if x['PROTOCOL_TESTING_MONTH'] == 'M3' and x['RISK_LEVEL'] == 'LR'
                                      else 9 if x['PROTOCOL_TESTING_MONTH'] == 'M4' and x['RISK_LEVEL'] == 'HR'
                                      else 10 if x['PROTOCOL_TESTING_MONTH'] == 'M4' and x['RISK_LEVEL'] == 'MR'
                                      else 11 if x['PROTOCOL_TESTING_MONTH'] == 'M4' and x['RISK_LEVEL'] == 'LR'
                                      else 12 if x['PROTOCOL_TESTING_MONTH'] == 'M6' and x['RISK_LEVEL'] == 'HR'
                                      else 13 if x['PROTOCOL_TESTING_MONTH'] == 'M6' and x['RISK_LEVEL'] == 'MR'
                                      else 14 if x['PROTOCOL_TESTING_MONTH'] == 'M6' and x['RISK_LEVEL'] == 'LR'
                                      else 15 if x['PROTOCOL_TESTING_MONTH'] == 'M9' and x['RISK_LEVEL'] == 'HR'
                                      else 16 if x['PROTOCOL_TESTING_MONTH'] == 'M9' and x['RISK_LEVEL'] == 'MR'
                                      else 17 if x['PROTOCOL_TESTING_MONTH'] == 'M9' and x['RISK_LEVEL'] == 'LR'
                                      else 18 if x['PROTOCOL_TESTING_MONTH'] == 'M12' and x['RISK_LEVEL'] == 'HR'
                                      else 19 if x['PROTOCOL_TESTING_MONTH'] == 'M12' and x['RISK_LEVEL'] == 'MR'
                                      else 20 if x['PROTOCOL_TESTING_MONTH'] == 'M12' and x['RISK_LEVEL'] == 'LR'
                                      else None, axis = 1)
sankey_df.head()
sankey_df['SANKEY'].nunique()

# Group by PID, PROTOCOL_TESTING_MONTH, RISK_LEVEL, and count unique accession into new dataframe
# Sort by Testing Days
sankey_df = sankey_df.groupby(['PID', 'PROTOCOL_TESTING_MONTH', 'RISK_LEVEL', 'SANKEY']).agg({'ACCESSIONID': 'nunique'}).reset_index()
sankey_df = sankey_df.sort_values(by = ['PID', 'SANKEY'], ascending = [True, True])

# Create a source and target column for the sankey diagram
sankey_df['SOURCE'] = sankey_df['SANKEY']
sankey_df['TARGET'] = sankey_df.groupby('PID')['SANKEY'].shift(-1)

# Drop rows where TARGET is NaN (last row in each patient)
df_links = sankey_df.dropna(subset=['TARGET']).copy()

# Rearranging and converting TARGET to int
df_links['TARGET'] = df_links['TARGET'].astype(int)
df_links = df_links[['PID', 'SOURCE', 'TARGET', 'ACCESSIONID']]


# Helper to convert hex to rgba with alpha
def hex_to_rgba(hex_color, alpha):
    rgba = mcolors.to_rgba(hex_color, alpha)
    return f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'

# Define node labels and colors
labels = ['HR M1', 'MR M1', 'LR M1', 
          'HR M2', 'MR M2', 'LR M2', 
          'HR M3', 'MR M3', 'LR M3', 
          'HR M4', 'MR M4', 'LR M4', 
          'HR M6', 'MR M6', 'LR M6', 
          'HR M9', 'MR M9', 'LR M9', 
          'HR M12', 'MR M12', 'LR M12']

# Assign node colors (solid for nodes)
node_colors = ['#d62728', '#ff7f0e', '#2ca02c'] * 7 

# Map source index to node color and lighten it for links
source = df_links['SOURCE'].values[:-1]
target = df_links['TARGET'].values[:-1]
value = df_links['ACCESSIONID'].values[:-1]

# Set alpha for light link appearance
link_alpha = 0.5
link_colors = [hex_to_rgba(node_colors[s], link_alpha) for s in source]

# Create Sankey figure
fig = go.Figure(go.Sankey(
    node = dict(pad = 50,
                thickness = 15,
                line = dict(color = "black", width = 0.3),
                label = labels,
                color = node_colors,
                x = [0.1, 0.1, 0.1, 
                     0.2, 0.2, 0.2,
                     0.3, 0.3, 0.3,
                     0.4, 0.4, 0.4,
                     0.6, 0.6, 0.6,
                     0.8, 0.8, 0.8,
                     1.0, 1.0, 1.0],
                y = [0.1, 0.25, 1,
                     0.1, 0.25, 1,
                     0.1, 0.25, 1,
                     0.1, 0.25, 1,
                     0.1, 0.25, 1,
                     0.1, 0.25, 1,
                     0.1, 0.25, 1],),
    link = dict(source = source,
                target = target,
                value = value,
                color = link_colors)))  # Now uses light rgba links

fig.update_layout(title_text="First Year AlloSure Testing Flow",  
                  font_size=15,
                  width=1300,
                  height=800,
                  margin=dict(t=40, b=270, l=10, r=10))

fig.show()




