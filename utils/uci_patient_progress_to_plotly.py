
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_patient_progress(first_year_df):
    """
    Generate violin plot showing AlloSure results across Low/Moderate/High Risk groups over time.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        plotly.graph_objects.Figure: Interactive violin plot showing risk group progression
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for HR (initial result >= 1% ) with atleast 2 tests results
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    # Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    # Filter for LR (initial result < 0.5%) with atleast 2 tests results
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()

    # Create stacked dataframe
    low_risk_df['RISK_DF'] = 'Low Risk'
    moderate_risk_df['RISK_DF'] = 'Moderate Risk'
    high_risk_df['RISK_DF'] = 'High Risk'

    stacked_df = pd.concat([low_risk_df, moderate_risk_df, high_risk_df], ignore_index=True)

    # Filter for less than 16
    stacked_df = stacked_df[stacked_df['RES_RESULT_NUMERIC'] < 16]

    # Create Plotly violin plot
    fig = px.violin(
        stacked_df, 
        x="PROTOCOL_TESTING_MONTH", 
        y="RES_RESULT_NUMERIC", 
        color="RISK_DF",
        title="AlloSure Results Across Low/Moderate/High Risk Groups",
        labels={
            "PROTOCOL_TESTING_MONTH": "Protocol Testing Month",
            "RES_RESULT_NUMERIC": "AlloSure (% dd-cfDNA)",
            "RISK_DF": "Risk Group"
        }
    )

    # Add horizontal reference lines
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="darkred",
        line_width=1
    )
    
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="orange",
        line_width=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='AlloSure Results Across Low/Moderate/High Risk Groups',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Protocol Testing Month',
        yaxis_title='AlloSure (% dd-cfDNA)',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickmode='auto',
            nticks=10
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            title='Risk Group',
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        width=1200,
        height=500,
        hovermode='closest'
    )

    # Update violin plot styling
    fig.update_traces(
        box_visible=True,
        meanline_visible=True,
        line_color='black',
        line_width=1
    )

    return fig


def get_patient_progress_summary(first_year_df):
    """
    Generate summary statistics for patient progress across risk groups.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        dict: Summary statistics including patient counts and risk group breakdown
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for HR (initial result >= 1% ) with atleast 2 tests results
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    # Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    # Filter for LR (initial result < 0.5%) with atleast 2 tests results
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()

    # Create summary statistics
    summary = {
        'total_patients': len(first_year_df['PID'].unique()),
        'high_risk_patients': len(high_risk_df['PID'].unique()),
        'moderate_risk_patients': len(moderate_risk_df['PID'].unique()),
        'low_risk_patients': len(low_risk_df['PID'].unique()),
        'high_risk_data': high_risk_df,
        'moderate_risk_data': moderate_risk_df,
        'low_risk_data': low_risk_df
    }
    
    return summary