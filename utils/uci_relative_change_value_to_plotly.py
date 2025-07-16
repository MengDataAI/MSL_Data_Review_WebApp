# utils/uci_relative_change_value_to_plotly.py

import numpy as np
import plotly.graph_objects as go
import pandas as pd

def plot_relative_change_value(first_year_df):
    rcv_df = first_year_df.copy()
    rcv_df['TEST_NUMBER'] = rcv_df.groupby('PID').cumcount() + 1
    rcv_df['RELATIVE_BETWEEN_TEST'] = round(rcv_df.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)
    rcv_plot = rcv_df.loc[rcv_df['RELATIVE_BETWEEN_TEST'].notna()].copy()
    rcv_plot['RCV'] = rcv_plot['RELATIVE_BETWEEN_TEST'].apply(
        lambda x: '≥253' if x >= 253 
        else ('≥149' if x >= 149 
              else ('≥61' if x >= 61 
                    else '<61'))
    )
    # Create summary table
    rcv_summary = rcv_plot.groupby('RCV').agg(
        N=('PID', 'nunique'),
        Median=('RELATIVE_BETWEEN_TEST', 'median'),
        Min=('RELATIVE_BETWEEN_TEST', 'min'),
        Max=('RELATIVE_BETWEEN_TEST', 'max')
    ).reset_index()
    rcv_summary['Median'] = rcv_summary['Median'].round(2)
    rcv_summary['Min'] = rcv_summary['Min'].round(2)
    rcv_summary['Max'] = rcv_summary['Max'].round(2)

    # Create Plotly figure
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=rcv_plot['RELATIVE_BETWEEN_TEST'],
        nbinsx=60,
        marker_color='black',
        opacity=0.8,
        marker_line_color='white',
        marker_line_width=0.8,
        name='Frequency'
    ))
    
    # Add background shading for risk categories
    min_val = rcv_plot['RELATIVE_BETWEEN_TEST'].min()
    max_val = rcv_plot['RELATIVE_BETWEEN_TEST'].max()
    
    # Green background (low risk: <61)
    fig.add_vrect(
        x0=min_val, x1=61,
        fillcolor='green',
        opacity=0.2,
        layer="below",
        line_width=0
    )
    
    # Yellow background (moderate risk: 61-149)
    fig.add_vrect(
        x0=61, x1=149,
        fillcolor='yellow',
        opacity=0.2,
        layer="below",
        line_width=0
    )
    
    # Red background (high risk: >149)
    fig.add_vrect(
        x0=149, x1=max_val,
        fillcolor='red',
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    # Add vertical lines at thresholds
    fig.add_vline(
        x=61,
        line_dash="dash",
        line_color="green",
        line_width=1
    )
    
    fig.add_vline(
        x=149,
        line_dash="dash",
        line_color="yellow",
        line_width=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Distribution of Relative Change in Between Tests',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Relative Change Between Sequential Tests (%)',
        yaxis_title='Frequency',
        xaxis=dict(
            range=[min_val, max_val],
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        width=1000,
        height=400
    )
    
    return fig, rcv_summary


def get_rcv_summary(first_year_df):
    """
    Generate summary statistics for Relative Change Value analysis.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        pd.DataFrame: Summary table with RCV categories and statistics
    """
    rcv_df = first_year_df.copy()
    rcv_df['TEST_NUMBER'] = rcv_df.groupby('PID').cumcount() + 1
    rcv_df['RELATIVE_BETWEEN_TEST'] = round(rcv_df.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)
    rcv_plot = rcv_df.loc[rcv_df['RELATIVE_BETWEEN_TEST'].notna()].copy()
    rcv_plot['RCV'] = rcv_plot['RELATIVE_BETWEEN_TEST'].apply(
        lambda x: '≥253' if x >= 253 
        else ('≥149' if x >= 149 
              else ('≥61' if x >= 61 
                    else '<61'))
    )
    
    # Create summary table
    rcv_summary = rcv_plot.groupby('RCV').agg(
        N=('PID', 'nunique'),
        Median=('RELATIVE_BETWEEN_TEST', 'median'),
        Min=('RELATIVE_BETWEEN_TEST', 'min'),
        Max=('RELATIVE_BETWEEN_TEST', 'max')
    ).reset_index()
    rcv_summary['Median'] = rcv_summary['Median'].round(2)
    rcv_summary['Min'] = rcv_summary['Min'].round(2)
    rcv_summary['Max'] = rcv_summary['Max'].round(2)
    
    return rcv_summary










