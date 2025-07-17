# utils/uci_patient_progress_high_risk_to_plotly.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_high_risk_progress(first_year_df):
    """
    Generate violin plot showing AlloSure results for High Risk patients over time.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        plotly.graph_objects.Figure: Interactive violin plot showing high risk patient progression
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for HR (initial result >= 1% ) with atleast 2 tests results
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    # Create Plotly violin plot
    fig = px.violin(
        high_risk_df, 
        x="PROTOCOL_TESTING_MONTH", 
        y="RES_RESULT_NUMERIC",
        title="High Risk Patients with Initial AlloSure Score ≥ 1%",
        labels={
            "PROTOCOL_TESTING_MONTH": "Protocol Testing Month",
            "RES_RESULT_NUMERIC": "AlloSure (% dd-cfDNA)"
        },
        category_orders={"PROTOCOL_TESTING_MONTH": ["M1", "M2", "M3", "M4", "M6", "M9", "M12"]}
    )

    # Add horizontal reference line at y=1
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="darkred",
        line_width=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='High Risk Patients with Initial AlloSure Score ≥ 1%',
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
        showlegend=False,
        width=900,
        height=500,
        hovermode='closest'
    )

    # Update violin plot styling
    fig.update_traces(
        box_visible=True,
        meanline_visible=True,
        line_color='black',
        line_width=1,
        fillcolor='lightcoral',
        opacity=0.7
    )

    return fig


def get_high_risk_summary(first_year_df):
    """
    Generate summary statistics for High Risk patients.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        dict: Summary statistics including patient count and data
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for HR (initial result >= 1% ) with atleast 2 tests results
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    # Create summary statistics
    summary = {
        'total_high_risk_patients': len(high_risk_df['PID'].unique()),
        'total_high_risk_records': len(high_risk_df),
        'high_risk_data': high_risk_df,
        'average_initial_score': high_risk_df.groupby('PID')['RES_RESULT_NUMERIC'].first().mean(),
        'median_initial_score': high_risk_df.groupby('PID')['RES_RESULT_NUMERIC'].first().median()
    }
    
    return summary


def get_high_risk_summary_table(first_year_df):
    """
    Generate detailed summary statistics table for High Risk patients by protocol testing month.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        pd.DataFrame: Rotated summary statistics table with N, Median (IQR), and Range by month
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for HR (initial result >= 1% ) with atleast 2 tests results
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    # AlloSure Results Summary Statistics - HR
    summary_stats_hr = high_risk_df.groupby('PROTOCOL_TESTING_MONTH').agg(
        N=('ACCESSIONID', 'nunique'),
        Median=("RES_RESULT_NUMERIC", "median"),
        Min=("RES_RESULT_NUMERIC", "min"),
        Max=("RES_RESULT_NUMERIC", "max"),
        p_25=("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.25)),
        p_75=("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.75))
    ).reset_index()

    summary_stats_hr['p_25'] = summary_stats_hr['p_25'].round(2)
    summary_stats_hr['p_75'] = summary_stats_hr['p_75'].round(2)
    summary_stats_hr['Min'] = summary_stats_hr['Min'].round(2)
    summary_stats_hr['Max'] = summary_stats_hr['Max'].round(2)
    summary_stats_hr['Median'] = summary_stats_hr['Median'].round(2)

    summary_stats_hr['Median (IQR)'] = (summary_stats_hr['Median'].astype(str) + " (" + 
                                        summary_stats_hr['p_25'].astype(str) + ", " + 
                                        summary_stats_hr['p_75'].astype(str) + ")")
    summary_stats_hr['Range'] = (summary_stats_hr['Min'].astype(str) + ", " + 
                                summary_stats_hr['Max'].astype(str))

    # Drop columns 
    summary_stats_hr.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)

    # Order protocol months 
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']

    first_col_hr = summary_stats_hr.columns[0]
    summary_stats_hr[first_col_hr] = pd.Categorical(summary_stats_hr[first_col_hr], categories=order, ordered=True)
    summary_stats_hr.sort_values(first_col_hr, inplace=True, ignore_index=True)

    # Rotate table
    rotated_hr = (summary_stats_hr.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(
        columns={'index': 'Protocol Testing Month'})

    return rotated_hr