# utils/uci_patient_progress_moderate_risk_to_plotly.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Generate violin plot showing AlloSure results for Moderate Risk patients over time.
def plot_moderate_risk_violin(first_year_df):
    """
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
    Returns:
        plotly.graph_objects.Figure: Interactive violin plot showing moderate risk patient progression
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    # Use a categorical color palette similar to Seaborn's default
    color_seq = px.colors.qualitative.Set2  # or Set1, Pastel, etc.
    month_order = ["M1", "M2", "M3", "M4", "M6", "M9", "M12"]

    # Create Plotly violin plot with each month a different color
    fig = px.violin(
        moderate_risk_df, 
        x="PROTOCOL_TESTING_MONTH", 
        y="RES_RESULT_NUMERIC",
        color="PROTOCOL_TESTING_MONTH",
        title="Moderate Risk Patients with Initial AlloSure Score ≥ 0.5 - < 1%",
        labels={
            "PROTOCOL_TESTING_MONTH": "Protocol Testing Month",
            "RES_RESULT_NUMERIC": "AlloSure (% dd-cfDNA)"
        },
        category_orders={"PROTOCOL_TESTING_MONTH": month_order},
        color_discrete_sequence=color_seq
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
            text='Moderate Risk Patients with Initial AlloSure Score ≥ 0.5 - < 1%',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Protocol Testing Month',
        yaxis_title='AlloSure (% dd-cfDNA)',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            categoryorder='array',
            categoryarray=month_order
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
        opacity=0.7
    )

    return fig


# Generate line plot showing Moderate Risk patients with subsequent AlloSure score ≥ 1%.
def plot_moderate_risk_lines(first_year_df):
    """
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
    Returns:
        plotly.graph_objects.Figure: Interactive line plot showing patient progression
    """
    # Global logic applied to all risk groups 
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    # Add test number
    moderate_risk_df['TEST_NUMBER'] = moderate_risk_df.groupby(['PID']).cumcount() + 1

    # Find moderate risk patients with a subsequent test >= 1%
    moderate_risk_subsequent_elevated = moderate_risk_df.loc[(moderate_risk_df['TEST_NUMBER'] != 1) & (moderate_risk_df['RES_RESULT_NUMERIC'] >= 1)]['PID'].nunique()
    moderate_risk_unique_pid = moderate_risk_df['PID'].nunique()
    moderate_risk_subsequent = round((moderate_risk_subsequent_elevated / moderate_risk_unique_pid) * 100, 1) if moderate_risk_unique_pid > 0 else 0

    # Get list of PIDs with a subsequent elevated result
    pid_filter_moderate = moderate_risk_df.loc[(moderate_risk_df['TEST_NUMBER'] != 1) & (moderate_risk_df['RES_RESULT_NUMERIC'] >= 1), 'PID']
    pid_filter_moderate = pd.DataFrame(pid_filter_moderate)
    moderate_risk_elevated_df = moderate_risk_df.loc[moderate_risk_df['PID'].isin(pid_filter_moderate['PID'])].copy()

    # Calculate Average Time to Initial Elevated Result
    first_test_moderate = moderate_risk_elevated_df.loc[moderate_risk_elevated_df['TEST_NUMBER'] == 1][['PID', 'RESULT_POST_TRANSPLANT']].rename(columns={'RESULT_POST_TRANSPLANT': 'START_DAY'})
    as_100_moderate = moderate_risk_elevated_df.loc[moderate_risk_elevated_df['RES_RESULT_NUMERIC'] >= 1].groupby('PID').first().reset_index()
    result_moderate = pd.merge(as_100_moderate[['PID', 'RESULT_POST_TRANSPLANT']], first_test_moderate, on='PID')
    result_moderate['TIME_TO_ELEVATED'] = result_moderate['RESULT_POST_TRANSPLANT'] - result_moderate['START_DAY']

    # Calculate Statistics
    mean_time_to_elevated_moderate = result_moderate['TIME_TO_ELEVATED'].mean()
    median_time_to_elevated_moderate = result_moderate['TIME_TO_ELEVATED'].median()
    ste_time_to_elevated_moderate = result_moderate['TIME_TO_ELEVATED'].sem()
    range_min_moderate = result_moderate['TIME_TO_ELEVATED'].quantile(0.25)
    range_max_moderate = result_moderate['TIME_TO_ELEVATED'].quantile(0.75)

    # Create Plotly figure
    fig = go.Figure()

    # Add line traces for each patient
    for patient_id in moderate_risk_elevated_df['PID'].unique():
        patient_data = moderate_risk_elevated_df[moderate_risk_elevated_df['PID'] == patient_id]
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

    # Add summary statistics as annotations
    fig.add_annotation(
        x=0.99, y=0.98,
        xref="paper", yref="paper",
        text=f"Average Time to Elevated Result (STE): {mean_time_to_elevated_moderate:.0f} ({ste_time_to_elevated_moderate:.0f}) days",
        showarrow=False,
        font=dict(size=9),
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0.99, y=0.94,
        xref="paper", yref="paper",
        text=f"Median (IQR): {median_time_to_elevated_moderate:.0f} ({range_min_moderate:.0f} - {range_max_moderate:.0f}) days",
        showarrow=False,
        font=dict(size=9),
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='Moderate Risk Patients with Subsequent AlloSure Score ≥ 1%',
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


# Generate detailed summary statistics table for Moderate Risk patients by protocol testing month.
def get_moderate_risk_summary_table(first_year_df):
    """
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

    # Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    # AlloSure Results Summary Statistics - MR
    summary_stats_mr = moderate_risk_df.groupby('PROTOCOL_TESTING_MONTH').agg(
        N=('ACCESSIONID', 'nunique'),
        Median=("RES_RESULT_NUMERIC", "median"),
        Min=("RES_RESULT_NUMERIC", "min"),
        Max=("RES_RESULT_NUMERIC", "max"),
        p_25=("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.25)),
        p_75=("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.75))
    ).reset_index()

    summary_stats_mr['p_25'] = summary_stats_mr['p_25'].round(2)
    summary_stats_mr['p_75'] = summary_stats_mr['p_75'].round(2)
    summary_stats_mr['Min'] = summary_stats_mr['Min'].round(2)
    summary_stats_mr['Max'] = summary_stats_mr['Max'].round(2)
    summary_stats_mr['Median'] = summary_stats_mr['Median'].round(2)

    summary_stats_mr['Median (IQR)'] = (summary_stats_mr['Median'].astype(str) + " (" + 
                                        summary_stats_mr['p_25'].astype(str) + ", " + 
                                        summary_stats_mr['p_75'].astype(str) + ")")
    summary_stats_mr['Range'] = (summary_stats_mr['Min'].astype(str) + ", " + 
                                summary_stats_mr['Max'].astype(str))

    # Drop columns 
    summary_stats_mr.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)

    # Order protocol months 
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']

    first_col_mr = summary_stats_mr.columns[0]
    summary_stats_mr[first_col_mr] = pd.Categorical(summary_stats_mr[first_col_mr], categories=order, ordered=True)
    summary_stats_mr.sort_values(first_col_mr, inplace=True, ignore_index=True)

    # Rotate table
    rotated_mr = (summary_stats_mr.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(
        columns={'index': 'Protocol Testing Month'})

    return rotated_mr




