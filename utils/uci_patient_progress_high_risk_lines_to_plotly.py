import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_high_risk_lines(first_year_df):
    """
    Plot high risk patients (initial AlloSure ≥ 1%) who had a subsequent test also ≥ 1%.
    Each patient is a line with X markers, alternating shaded intervals for months, and a horizontal line at y=1.
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
    Returns:
        plotly.graph_objects.Figure: Interactive line plot
    """
    # Sort and group
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for HR (initial result >= 1% ) with at least 2 tests
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    # Add test number
    high_risk_df['TEST_NUMBER'] = high_risk_df.groupby(['PID']).cumcount() + 1

    # Find high risk patients with a subsequent test >= 1%
    high_risk_subsequent_elevated = high_risk_df.loc[(high_risk_df['TEST_NUMBER'] != 1) & (high_risk_df['RES_RESULT_NUMERIC'] >= 1)]['PID'].nunique()
    high_risk_unique_pid = high_risk_df['PID'].nunique()
    high_risk_subsequent = round((high_risk_subsequent_elevated / high_risk_unique_pid) * 100, 1) if high_risk_unique_pid > 0 else 0

    # Get list of PIDs with a subsequent elevated result
    pid_filter_high = high_risk_df.loc[(high_risk_df['TEST_NUMBER'] != 1) & (high_risk_df['RES_RESULT_NUMERIC'] >= 1), 'PID']
    pid_filter_high = pd.DataFrame(pid_filter_high)
    high_risk_elevated_df = high_risk_df.loc[high_risk_df['PID'].isin(pid_filter_high['PID'])].copy()

    # Create Plotly figure
    fig = go.Figure()

    # Add line traces for each patient
    for patient_id in high_risk_elevated_df['PID'].unique():
        patient_data = high_risk_elevated_df[high_risk_elevated_df['PID'] == patient_id]
        fig.add_trace(go.Scatter(
            x=patient_data['RESULT_POST_TRANSPLANT'],
            y=patient_data['RES_RESULT_NUMERIC'],
            mode='lines+markers',
            name=f'Patient {patient_id}',
            line=dict(width=0.8),
            marker=dict(symbol='x', size=8, opacity=0.8),
            hovertemplate='<b>Patient %{fullData.name}</b><br>' +
                         'Days Post-Transplant: %{x}<br>' +
                         'AlloSure Score: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

    # Add alternating background shading for intervals
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
            text='High Risk Patients with Subsequent AlloSure Score ≥ 1%',
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
        hovermode='closest',
        margin=dict(l=60, r=20, t=60, b=60)
    )

    return fig



