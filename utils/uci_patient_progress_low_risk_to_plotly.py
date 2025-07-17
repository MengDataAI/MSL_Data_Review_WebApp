import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from lifelines import CoxPHFitter, KaplanMeierFitter

# Generate detailed summary statistics table for Low Risk patients by protocol testing month.
def get_low_risk_summary_table(first_year_df):
    """
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with PROTOCOL_TESTING_MONTH
    Returns:
        pd.DataFrame: Rotated summary statistics table with N, Median (IQR), and Range by month
    """
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()
    summary_stats_lr = low_risk_df.groupby('PROTOCOL_TESTING_MONTH').agg(
        N=('ACCESSIONID', 'nunique'),
        Median=("RES_RESULT_NUMERIC", "median"),
        Min=("RES_RESULT_NUMERIC", "min"),
        Max=("RES_RESULT_NUMERIC", "max"),
        p_25=("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.25)),
        p_75=("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.75))
    ).reset_index()
    summary_stats_lr['p_25'] = summary_stats_lr['p_25'].round(2)
    summary_stats_lr['p_75'] = summary_stats_lr['p_75'].round(2)
    summary_stats_lr['Min'] = summary_stats_lr['Min'].round(2)
    summary_stats_lr['Max'] = summary_stats_lr['Max'].round(2)
    summary_stats_lr['Median'] = summary_stats_lr['Median'].round(2)
    summary_stats_lr['Median (IQR)'] = (summary_stats_lr['Median'].astype(str) + " (" + summary_stats_lr['p_25'].astype(str) + ", " + summary_stats_lr['p_75'].astype(str) + ")")
    summary_stats_lr['Range'] = (summary_stats_lr['Min'].astype(str) + ", " + summary_stats_lr['Max'].astype(str))
    summary_stats_lr.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)
    order = ["M1", "M2", "M3", "M4", "M6", "M9", "M12"]
    first_col_lr = summary_stats_lr.columns[0]
    summary_stats_lr[first_col_lr] = pd.Categorical(summary_stats_lr[first_col_lr], categories=order, ordered=True)
    summary_stats_lr.sort_values(first_col_lr, inplace=True, ignore_index=True)
    rotated_lr = (summary_stats_lr.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(columns={'index': 'Protocol Testing Month'})
    return rotated_lr

# Generate Cox proportional hazards summary table
def generate_cox_table(first_year_df, duration_col, event_col, covariates):
    """
    Fit a Cox proportional hazards model and return summary table.
    Args:
        first_year_df (pd.DataFrame): DataFrame with survival data
        duration_col (str): Column name for time-to-event
        event_col (str): Column name for event/censoring (1=event, 0=censored)
        covariates (list): List of covariate column names
    Returns:
        pd.DataFrame: Cox model summary (HR, CI, p-value)
    """
    df = first_year_df[[duration_col, event_col] + covariates].dropna()
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    summary = cph.summary.copy()
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_lower'] = np.exp(summary['coef lower 95%'])
    summary['CI_upper'] = np.exp(summary['coef upper 95%'])
    return summary[['HR', 'CI_lower', 'CI_upper', 'p']]

# Generate forest plot for Cox model results
def plot_forest_cox(first_year_df, duration_col, event_col, covariates):
    """
    Fit a Cox model and plot a forest plot of HRs and CIs.
    Args:
        first_year_df (pd.DataFrame): DataFrame with survival data
        duration_col (str): Column name for time-to-event
        event_col (str): Column name for event/censoring (1=event, 0=censored)
        covariates (list): List of covariate column names
    Returns:
        plotly.graph_objects.Figure: Forest plot
    """
    summary = generate_cox_table(first_year_df, duration_col, event_col, covariates)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary['HR'],
        y=summary.index,
        error_x=dict(
            type='data',
            symmetric=False,
            array=summary['CI_upper'] - summary['HR'],
            arrayminus=summary['HR'] - summary['CI_lower']
        ),
        mode='markers',
        marker=dict(color='darkblue', size=12),
        orientation='h',
        name='Hazard Ratio'
    ))
    fig.add_vline(x=1, line_dash='dash', line_color='gray')
    fig.update_layout(
        title='Cox Proportional Hazards Model: Forest Plot',
        xaxis_title='Hazard Ratio (log scale)',
        yaxis_title='Covariate',
        xaxis_type='log',
        template='simple_white',
        height=400,
        width=700
    )
    return fig

# Generate Kaplan-Meier plot for two groups
def plot_km_two_groups(first_year_df, duration_col, event_col, group_col, group_labels=None):
    """
    Plot KM curve for two groups.
    Args:
        first_year_df (pd.DataFrame): DataFrame with survival data
        duration_col (str): Column name for time-to-event
        event_col (str): Column name for event/censoring (1=event, 0=censored)
        group_col (str): Column name for group (must have 2 unique values)
        group_labels (list): Optional, labels for the two groups
    Returns:
        plotly.graph_objects.Figure: KM plot
    """
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    groups = first_year_df[group_col].dropna().unique()
    if group_labels is None:
        group_labels = groups
    colors = px.colors.qualitative.Set1
    for i, group in enumerate(groups):
        mask = first_year_df[group_col] == group
        kmf.fit(first_year_df.loc[mask, duration_col], first_year_df.loc[mask, event_col], label=str(group_labels[i]))
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_[kmf._label],
            mode='lines',
            name=str(group_labels[i]),
            line=dict(color=colors[i % len(colors)], width=3)
        ))
    fig.update_layout(
        title='Kaplan-Meier Survival Curve (2 Groups)',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        template='simple_white',
        width=700,
        height=400
    )
    return fig

# Generate Kaplan-Meier plot for multiple groups
def plot_km_multi_groups(first_year_df, duration_col, event_col, group_col):
    """
    Plot KM curves for multiple groups.
    Args:
        first_year_df (pd.DataFrame): DataFrame with survival data
        duration_col (str): Column name for time-to-event
        event_col (str): Column name for event/censoring (1=event, 0=censored)
        group_col (str): Column name for group (can have >2 unique values)
    Returns:
        plotly.graph_objects.Figure: KM plot
    """
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    groups = first_year_df[group_col].dropna().unique()
    colors = px.colors.qualitative.Set2
    for i, group in enumerate(groups):
        mask = first_year_df[group_col] == group
        kmf.fit(first_year_df.loc[mask, duration_col], first_year_df.loc[mask, event_col], label=str(group))
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_[kmf._label],
            mode='lines',
            name=str(group),
            line=dict(color=colors[i % len(colors)], width=3)
        ))
    fig.update_layout(
        title='Kaplan-Meier Survival Curve (Multiple Groups)',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        template='simple_white',
        width=800,
        height=400
    )
    return fig







