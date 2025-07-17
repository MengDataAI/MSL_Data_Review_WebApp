




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