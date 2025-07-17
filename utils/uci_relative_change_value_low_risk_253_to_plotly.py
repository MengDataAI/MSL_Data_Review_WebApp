
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_low_risk_253(first_year_df):
    """
    Generate line plot for Low Risk patients with RCV > 253%.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        plotly.graph_objects.Figure: Interactive line plot showing patient progression
    """
    # Sort by Patient ID and days post-transplant
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Create Low Risk df
    # Filter for LR (initial result < 0.5%) with atleast 2 tests results
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()

    # Modify for the Low Risk Group
    rcv_df_low = low_risk_df.copy()
    rcv_df_low['TEST_NUMBER'] = rcv_df_low.groupby('PID').cumcount() + 1

    # Calculate Relative Between Tests for Each Patient
    rcv_df_low['RELATIVE_BETWEEN_TEST'] = round(rcv_df_low.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

    # Create RCV dataframe - exclude NA (first result)
    rcv_plot_low = rcv_df_low.loc[rcv_df_low['RELATIVE_BETWEEN_TEST'].notna()].copy()

    # Calculate the percent of RELATIVE_BETWEEN_TEST >=253, >=149, >=61 AND <149, <61
    rcv_plot_low['RCV'] = rcv_plot_low['RELATIVE_BETWEEN_TEST'].apply(
        lambda x: '≥253' if x >= 253 
        else ('≥149' if x >= 149 
              else ('≥61' if x >= 61 
                    else '<61'))
    )

    # Create Dataframe for the Patients with change > 253%
    high_change_patients_low = rcv_plot_low[rcv_plot_low['RELATIVE_BETWEEN_TEST'] > 253]['PID'].unique()

    # Filter for Patients
    high_change_patients_low_filtered_df = low_risk_df[low_risk_df['PID'].isin(high_change_patients_low)]

    # Create Plotly figure
    fig = go.Figure()

    # Add line traces for each patient
    for patient_id in high_change_patients_low_filtered_df['PID'].unique():
        patient_data = high_change_patients_low_filtered_df[high_change_patients_low_filtered_df['PID'] == patient_id]
        
        fig.add_trace(go.Scatter(
            x=patient_data['RESULT_POST_TRANSPLANT'],
            y=patient_data['RES_RESULT_NUMERIC'],
            mode='lines+markers',
            name=f'Patient {patient_id}',
            line=dict(width=0.8),
            marker=dict(
                symbol='x',
                size=8,
                opacity=0.8
            ),
            hovertemplate='<b>Patient %{fullData.name}</b><br>' +
                         'Days Post-Transplant: %{x}<br>' +
                         'AlloSure Score: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

    # Add alternating background shading
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    
    for i, (start, end) in enumerate(intervals):
        color = 'lightgray' if i % 2 == 0 else 'white'
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=0.35,
            layer="below",
            line_width=0
        )

    # Add horizontal reference lines
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="darkred",
        line_width=1,
        annotation_text="1% Threshold"
    )
    
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="orange",
        line_width=1,
        annotation_text="0.5% Threshold"
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='Low Risk Patients with Sequential AlloSure Score RCV >253%',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Days Post-Transplant',
        yaxis_title='AlloSure (% dd-cfDNA)',
        xaxis=dict(
            range=[0, 395],
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
            title='Patient ID',
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        width=1000,
        height=500,
        hovermode='closest'
    )

    return fig


def get_low_risk_253_summary(first_year_df):
    """
    Generate summary statistics for Low Risk patients with RCV > 253%.
    
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
        
    Returns:
        dict: Summary statistics including patient count and RCV breakdown
    """
    # Sort by Patient ID and days post-transplant
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Create Low Risk df
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()

    # Modify for the Low Risk Group
    rcv_df_low = low_risk_df.copy()
    rcv_df_low['TEST_NUMBER'] = rcv_df_low.groupby('PID').cumcount() + 1
    rcv_df_low['RELATIVE_BETWEEN_TEST'] = round(rcv_df_low.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)

    # Create RCV dataframe - exclude NA (first result)
    rcv_plot_low = rcv_df_low.loc[rcv_df_low['RELATIVE_BETWEEN_TEST'].notna()].copy()

    # Calculate the percent of RELATIVE_BETWEEN_TEST >=253, >=149, >=61 AND <149, <61
    rcv_plot_low['RCV'] = rcv_plot_low['RELATIVE_BETWEEN_TEST'].apply(
        lambda x: '≥253' if x >= 253 
        else ('≥149' if x >= 149 
              else ('≥61' if x >= 61 
                    else '<61'))
    )

    # Group By RCV and get percent total
    rcv_groupby_low = round(rcv_plot_low.groupby('RCV')['PID'].count()/len(rcv_plot_low)* 100,0)
    rcv_group_sample_size_low = rcv_plot_low.groupby('RCV')['PID'].count()

    # Order this table by <61, >=61, >=149, >=253
    rcv_groupby_low = rcv_groupby_low.reindex(['<61', '≥61', '≥149', '≥253'])
    rcv_group_sample_size_low = rcv_group_sample_size_low.reindex(['<61', '≥61', '≥149', '≥253'])

    # Change to string 
    rcv_groupby_low = rcv_groupby_low.astype(int).astype(str) + '%'
    rcv_group_sample_size_low = rcv_group_sample_size_low.fillna(0).astype(int)

    # Combined Table
    rcv_summary_low = pd.DataFrame({
        '(N)': rcv_group_sample_size_low,
        'RCV': rcv_groupby_low
    })

    rcv_summary_low.index.name = "RCV Range (%)"
    rcv_summary_low = rcv_summary_low.reset_index()

    # Get patients with RCV > 253%
    high_change_patients_low = rcv_plot_low[rcv_plot_low['RELATIVE_BETWEEN_TEST'] > 253]['PID'].unique()
    
    summary = {
        'total_low_risk_patients': low_risk_df['PID'].nunique(),
        'patients_with_rcv_gt_253': len(high_change_patients_low),
        'rcv_summary_table': rcv_summary_low,
        'high_change_patient_ids': list(high_change_patients_low)
    }
    
    return summary