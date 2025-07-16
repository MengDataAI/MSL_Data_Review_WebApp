# utils/uci_testing_pattern_to_plotly.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def get_first_year_df(spark):
    spark_df = spark.read.table("production.data_engineering.UCI_ANALYSIS_PIPELINE")
    df = spark_df.toPandas()
    first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()]
    return first_year_df

def plot_histogram(first_year_df):
    # Create the histogram
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=first_year_df['RESULT_POST_TRANSPLANT'],
        nbinsx=50,
        marker_color='darkblue',
        opacity=0.8,
        marker_line_color='black',
        marker_line_width=1,
        name='Frequency'
    ))
    
    # Add background shading for protocol months
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    colors = ['grey', 'white', 'grey', 'white', 'grey', 'white', 'grey']
    
    for i, ((start, end), color) in enumerate(zip(intervals, colors)):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=0.35,
            layer="below",
            line_width=0
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='AlloSure Testing Frequency',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Days Post-Transplant',
        yaxis_title='Frequency',
        xaxis=dict(
            range=[0, 395],
            dtick=14,
            tickmode='linear'
        ),
        showlegend=False,
        width=1000,
        height=400
    )
    
    return fig

def plot_barplot(first_year_df):
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    palette = px.colors.qualitative.Set3[:len(order)]  # Plotly color palette
    
    # Group and sort data
    grouped_testing_month = first_year_df.groupby('PROTOCOL_TESTING_MONTH').agg({
        'ACCESSIONID': 'nunique', 
        'RESULT_POST_TRANSPLANT': 'mean'
    }).reset_index()
    
    # Ensure proper ordering
    grouped_testing_month['PROTOCOL_TESTING_MONTH'] = pd.Categorical(
        grouped_testing_month['PROTOCOL_TESTING_MONTH'], 
        categories=order, 
        ordered=True
    )
    grouped_testing_month = grouped_testing_month.sort_values('PROTOCOL_TESTING_MONTH')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=grouped_testing_month['PROTOCOL_TESTING_MONTH'],
        y=grouped_testing_month['ACCESSIONID'],
        marker_color=palette,
        marker_line_color='black',
        marker_line_width=1,
        text=grouped_testing_month['ACCESSIONID'],
        textposition='outside',
        textfont=dict(color='white', size=10),
        name='Number of Tests'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Number of Tests by First Year Schedule',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Protocol Testing Month',
        yaxis_title='Number of Tests',
        showlegend=False,
        width=600,
        height=400
    )
    
    return fig

def plot_violin(first_year_df):
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    palette = px.colors.qualitative.Set3[:len(order)]  # Plotly color palette
    
    # Create violin plot
    fig = go.Figure()
    
    for i, month in enumerate(order):
        month_data = first_year_df[first_year_df['PROTOCOL_TESTING_MONTH'] == month]['RES_RESULT_NUMERIC']
        if not month_data.empty:
            fig.add_trace(go.Violin(
                x=[month] * len(month_data),
                y=month_data,
                name=month,
                line_color=palette[i],
                fillcolor=palette[i],
                opacity=0.7,
                showlegend=False
            ))
    
    # Add horizontal line at y=1
    fig.add_hline(
        y=1, 
        line_dash="dash", 
        line_color="darkred",
        line_width=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Distribution of AlloSure (% dd-cfDNA) by First Year Schedule',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Protocol Testing Month',
        yaxis_title='AlloSure (% dd-cfDNA)',
        xaxis=dict(categoryorder='array', categoryarray=order),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        width=800,
        height=400
    )
    
    return fig

def get_statistics_table(first_year_df):
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
    summary_stats_2b.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)
    order = [ 'M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    first_col = summary_stats_2b.columns[0]
    summary_stats_2b[first_col] = pd.Categorical(summary_stats_2b[first_col], categories=order, ordered=True)
    summary_stats_2b.sort_values(first_col, inplace=True, ignore_index=True)
    rotated = (summary_stats_2b.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(columns={'index': 'Protocol Testing Month'})
    return rotated