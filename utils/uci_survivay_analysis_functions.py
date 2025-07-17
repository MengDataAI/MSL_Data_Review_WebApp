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
def plot_forest_cox(first_year_df):
    """
    Fit a Cox model and plot a forest plot of HRs and CIs for risk factors.
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with required columns
    Returns:
        plotly.graph_objects.Figure: Forest plot
    """
    # Prepare data for survival analysis
    df = first_year_df[['PID', 'ACCESSIONID', 'RESULT_POST_TRANSPLANT', 'DONOR_TYPE', 'RES_RESULT_NUMERIC', 'ON_PROTOCOL', 'PROTOCOL_TESTING_MONTH', 'TRANSPLANT_AGE', 'SEX']].copy()
    
    # Convert Transplant Age to Int
    df['TRANSPLANT_AGE'] = df['TRANSPLANT_AGE'].astype(int)
    
    # Filter for only patients on protocol and for results in the first year
    df = df[(df['ON_PROTOCOL'] == 'Y') & (df['PROTOCOL_TESTING_MONTH'].notna())]
    
    # Order dataframe by PID and RESULT_POST_TRANSPLANT
    df = df.sort_values(by=['PID', 'RESULT_POST_TRANSPLANT'])
    
    # Add a Test Count Column
    df['TEST_NUMBER'] = df.groupby(['PID']).cumcount() + 1
    
    # Global logic applied to all risk groups 
    sorted_first_year_df = df.sort_values(['PID', 'TEST_NUMBER']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')
    
    # Moderate Risk
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    
    # Create a column in dataframe to tag mask_mr and mask_lr
    df['RISK_GROUP'] = np.where(mask_mr, 'MR', np.where(mask_lr, 'LR', 'EXCLUDE'))
    
    # Filter out Nan in Risk Group
    df = df[df['RISK_GROUP'] != 'EXCLUDE']
    
    # Create TIME_TO_EVENT Column as well as Binary classification for whether event happened
    df['TIME_TO_EVENT'] = None
    df['EVENT_OCCURRED'] = None
    
    for pid, patient_data in df.groupby('PID'):
        patient_data = patient_data.sort_values(['PID','RESULT_POST_TRANSPLANT'])
        over_cutoff = patient_data[patient_data['RES_RESULT_NUMERIC'] >= 1]
        
        if not over_cutoff.empty:
            # Look for the first time patient receive AS score >=1%
            time_to_event = over_cutoff.iloc[0]['RESULT_POST_TRANSPLANT']
            event_occurred = 1
        else:
            # If not available then take max result post transplant (last test), and event occurred = 0
            time_to_event = patient_data['RESULT_POST_TRANSPLANT'].max()
            event_occurred = 0
        
        df.loc[(df['PID'] == pid), 'TIME_TO_EVENT'] = time_to_event
        df.loc[(df['PID'] == pid), 'EVENT_OCCURRED'] = event_occurred
    
    # Convert to Int
    df['TIME_TO_EVENT'] = df['TIME_TO_EVENT'].astype(int)
    df['EVENT_OCCURRED'] = df['EVENT_OCCURRED'].astype(int)
    
    # One-hot encode DONOR_TYPE, RISK_GROUP, and SEX
    df_encoded = pd.get_dummies(df, columns=['DONOR_TYPE', 'RISK_GROUP', 'SEX'], drop_first=False)
    
    # Convert the one hot encoded columns to integer
    df_encoded[['DONOR_TYPE_Deceased', 'DONOR_TYPE_Living','RISK_GROUP_MR', 'RISK_GROUP_LR', 'SEX_Male', 'SEX_Female']] = df_encoded[['DONOR_TYPE_Deceased', 'DONOR_TYPE_Living','RISK_GROUP_MR', 'RISK_GROUP_LR', 'SEX_Male', 'SEX_Female']].astype(int)
    
    # Select columns for analysis and drop duplicates, reference is DONOR TYPE = Living, Low Risk Group, Sex Male
    df_encoded = df_encoded[['PID', 'TIME_TO_EVENT', 'EVENT_OCCURRED', 'DONOR_TYPE_Deceased', 'RISK_GROUP_MR', 'SEX_Female', 'TRANSPLANT_AGE']]
    df_encoded = df_encoded.drop_duplicates()
    
    df_encoded = df_encoded[['TIME_TO_EVENT', 'EVENT_OCCURRED', 'DONOR_TYPE_Deceased', 'RISK_GROUP_MR', 'SEX_Female', 'TRANSPLANT_AGE']]
    
    # Initialize and fit the model
    cph = CoxPHFitter()
    cph.fit(df_encoded, duration_col='TIME_TO_EVENT', event_col='EVENT_OCCURRED')
    
    # Get summary and calculate hazard ratios
    summary = cph.summary.copy()
    summary['logHR'] = summary['coef']
    summary['logCI_lower'] = summary['coef lower 95%']
    summary['logCI_upper'] = summary['coef upper 95%']
    summary['p_text'] = summary['p'].apply(lambda p: f"{p:.2e}" if p < 0.0001 else f"{p:.4f}")
    
    # Create forest plot
    fig = go.Figure()
    # Add confidence intervals as thick, semi-transparent lines
    fig.add_trace(go.Scatter(
        x=summary['logHR'],
        y=summary.index,
        error_x=dict(
            type='data',
            symmetric=False,
            array=summary['logCI_upper'] - summary['logHR'],
            arrayminus=summary['logHR'] - summary['logCI_lower'],
            thickness=8,
            color='rgba(30,144,255,0.25)'
        ),
        mode='markers',
        marker=dict(color='rgb(30,144,255)', size=18, line=dict(width=2, color='black'), symbol='circle'),
        orientation='h',
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>log(HR): %{x:.2f}<br>p: %{customdata}<extra></extra>',
        customdata=summary['p_text']
    ))
    # Add point estimate annotation for each covariate
    for idx, row in summary.iterrows():
        fig.add_annotation(
            x=row['logHR'],
            y=idx,
            text=f"log(HR): {row['logHR']:.2f}<br>p: {row['p_text']}",
            showarrow=False,
            font=dict(size=12, color='black'),
            xanchor='left',
            yanchor='middle',
            xshift=20,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    # Add bold, dashed reference line at log(HR)=0
    fig.add_vline(x=0, line_dash='dash', line_color='black', line_width=3)
    # Update layout for publication style
    fig.update_layout(
        title=dict(text='<b>Cox Proportional Hazards Model: Forest Plot</b>', x=0.5, xanchor='center', font=dict(size=22)),
        xaxis_title='<b>log(Hazard Ratio)</b>',
        yaxis_title='',
        xaxis_type='linear',
        xaxis=dict(range=[-4, 4], tickvals=[-4, -2, 0, 2, 4], tickfont=dict(size=16), gridcolor='lightgray', zeroline=False, showline=True, linewidth=2, linecolor='gray'),
        yaxis=dict(tickfont=dict(size=16, family='Arial', color='black'), showgrid=True, gridcolor='gainsboro', showline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=500,
        margin=dict(l=120, r=40, t=80, b=60),
        showlegend=False,
    )
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    return fig

# Generate Kaplan-Meier plot for two groups (Risk Groups)
def plot_km_moderate_low(first_year_df):
    """
    Plot KM curve for two risk groups (Moderate Risk vs Low Risk).
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with required columns
    Returns:
        plotly.graph_objects.Figure: KM plot comparing risk groups
    """
    # Prepare data for survival analysis
    df = first_year_df[['PID', 'ACCESSIONID', 'RESULT_POST_TRANSPLANT', 'DONOR_TYPE', 'RES_RESULT_NUMERIC', 'ON_PROTOCOL', 'PROTOCOL_TESTING_MONTH', 'TRANSPLANT_AGE', 'SEX']].copy()
    
    # Convert Transplant Age to Int
    df['TRANSPLANT_AGE'] = df['TRANSPLANT_AGE'].astype(int)
    
    # Filter for only patients on protocol and for results in the first year
    df = df[(df['ON_PROTOCOL'] == 'Y') & (df['PROTOCOL_TESTING_MONTH'].notna())]
    
    # Order dataframe by PID and RESULT_POST_TRANSPLANT
    df = df.sort_values(by=['PID', 'RESULT_POST_TRANSPLANT'])
    
    # Add a Test Count Column
    df['TEST_NUMBER'] = df.groupby(['PID']).cumcount() + 1
    
    # Global logic applied to all risk groups 
    sorted_first_year_df = df.sort_values(['PID', 'TEST_NUMBER']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')
    
    # Moderate Risk
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    
    # Create a column in dataframe to tag mask_mr and mask_lr
    df['RISK_GROUP'] = np.where(mask_mr, 'MR', np.where(mask_lr, 'LR', 'EXCLUDE'))
    
    # Filter out Nan in Risk Group
    df = df[df['RISK_GROUP'] != 'EXCLUDE']
    
    # Create TIME_TO_EVENT Column as well as Binary classification for whether event happened
    df['TIME_TO_EVENT'] = None
    df['EVENT_OCCURRED'] = None
    
    for pid, patient_data in df.groupby('PID'):
        patient_data = patient_data.sort_values(['PID','RESULT_POST_TRANSPLANT'])
        over_cutoff = patient_data[patient_data['RES_RESULT_NUMERIC'] >= 1]
        
        if not over_cutoff.empty:
            # Look for the first time patient receive AS score >=1%
            time_to_event = over_cutoff.iloc[0]['RESULT_POST_TRANSPLANT']
            event_occurred = 1
        else:
            # If not available then take max result post transplant (last test), and event occurred = 0
            time_to_event = patient_data['RESULT_POST_TRANSPLANT'].max()
            event_occurred = 0
        
        df.loc[(df['PID'] == pid), 'TIME_TO_EVENT'] = time_to_event
        df.loc[(df['PID'] == pid), 'EVENT_OCCURRED'] = event_occurred
    
    # Convert to Int
    df['TIME_TO_EVENT'] = df['TIME_TO_EVENT'].astype(int)
    df['EVENT_OCCURRED'] = df['EVENT_OCCURRED'].astype(int)
    
    # Dataframe for Kaplan Meier Curves
    km_df = df[['PID', 'TIME_TO_EVENT', 'EVENT_OCCURRED', 'RISK_GROUP']].copy()
    km_df = km_df.drop_duplicates()
    km_df = km_df.sort_values(by=['PID', 'TIME_TO_EVENT'])
    
    # Split into two groups
    group_MR = km_df[km_df['RISK_GROUP'] == 'MR']
    group_LR = km_df[km_df['RISK_GROUP'] == 'LR']
    
    # Initialize the KaplanMeierFitter
    kmf_mr = KaplanMeierFitter()
    kmf_lr = KaplanMeierFitter()
    
    # Fit models
    kmf_mr.fit(group_MR['TIME_TO_EVENT'], event_observed=group_MR['EVENT_OCCURRED'], label='Moderate Risk')
    kmf_lr.fit(group_LR['TIME_TO_EVENT'], event_observed=group_LR['EVENT_OCCURRED'], label='Low Risk')
    
    # Colors (colorblind-friendly)
    color_mr = 'rgb(230, 97, 1)'   # orange
    color_lr = 'rgb(94, 60, 153)'  # blue-purple
    ci_alpha = 0.18
    
    # Create Plotly figure
    fig = go.Figure()
    # Moderate Risk curve
    fig.add_trace(go.Scatter(
        x=kmf_mr.survival_function_.index,
        y=kmf_mr.survival_function_['Moderate Risk'],
        mode='lines',
        name='Moderate Risk',
        line=dict(color=color_mr, width=4),
        hoverinfo='x+y+name',
        showlegend=True
    ))
    # Moderate Risk CI
    fig.add_trace(go.Scatter(
        x=list(kmf_mr.confidence_interval_.index) + list(kmf_mr.confidence_interval_.index[::-1]),
        y=list(kmf_mr.confidence_interval_['Moderate Risk_upper_0.95']) + list(kmf_mr.confidence_interval_['Moderate Risk_lower_0.95'][::-1]),
        fill='toself',
        fillcolor=f'rgba(230,97,1,{ci_alpha})',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    # Low Risk curve
    fig.add_trace(go.Scatter(
        x=kmf_lr.survival_function_.index,
        y=kmf_lr.survival_function_['Low Risk'],
        mode='lines',
        name='Low Risk',
        line=dict(color=color_lr, width=4, dash='solid'),
        hoverinfo='x+y+name',
        showlegend=True
    ))
    # Low Risk CI
    fig.add_trace(go.Scatter(
        x=list(kmf_lr.confidence_interval_.index) + list(kmf_lr.confidence_interval_.index[::-1]),
        y=list(kmf_lr.confidence_interval_['Low Risk_upper_0.95']) + list(kmf_lr.confidence_interval_['Low Risk_lower_0.95'][::-1]),
        fill='toself',
        fillcolor=f'rgba(94,60,153,{ci_alpha})',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    # Calculate p-value using log-rank test
    from lifelines.statistics import logrank_test
    results = logrank_test(group_MR['TIME_TO_EVENT'], group_LR['TIME_TO_EVENT'], 
                          group_MR['EVENT_OCCURRED'], group_LR['EVENT_OCCURRED'])
    p_value = results.p_value
    # P-value annotation formatting
    if p_value < 0.0001:
        p_text = f'P-value: {p_value:.2e}'
    else:
        p_text = f'P-value: {p_value:.4f}'
    # Add p-value annotation (large, bold, with white background and border)
    fig.add_annotation(
        x=0.99, y=0.05, xref="paper", yref="paper",
        text=f'<b>{p_text}</b>',
        showarrow=False, font=dict(size=18, color='black', family='Arial'), align="right",
        bgcolor="white", bordercolor="black", borderwidth=2, borderpad=6
    )
    # Update layout for publication style
    fig.update_layout(
        title=dict(text='<b>Survival Curves Between Moderate and Low Risk Groups</b>', x=0.5, xanchor='center', font=dict(size=22)),
        xaxis_title='<b>Time (Days Post-Transplant)</b>',
        yaxis_title='<b>Free from AS>1%</b>',
        xaxis=dict(range=[0, 395], showgrid=False, showline=True, linewidth=2, linecolor='gray', tickfont=dict(size=16), titlefont=dict(size=18)),
        yaxis=dict(showgrid=True, gridcolor='gainsboro', zeroline=False, showline=True, linewidth=2, linecolor='gray', tickfont=dict(size=16), titlefont=dict(size=18)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=540,
        margin=dict(l=100, r=40, t=80, b=60),
        legend=dict(
            x=1.02, y=1, xanchor='left', yanchor='top',
            font=dict(size=16, family='Arial'),
            bordercolor='gray', borderwidth=1, bgcolor='rgba(255,255,255,0.8)'
        ),
        showlegend=True,
    )
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    return fig


# Generate Kaplan-Meier plot for donor type groups (Deceased vs Living)
def plot_km_donor_type(first_year_df):
    """
    Plot KM curve for donor type groups (Deceased vs Living).
    Args:
        first_year_df (pd.DataFrame): DataFrame containing patient data with required columns
    Returns:
        plotly.graph_objects.Figure: KM plot comparing donor types
    """
    # Prepare data for survival analysis
    df = first_year_df[['PID', 'ACCESSIONID', 'RESULT_POST_TRANSPLANT', 'DONOR_TYPE', 'RES_RESULT_NUMERIC', 'ON_PROTOCOL', 'PROTOCOL_TESTING_MONTH', 'TRANSPLANT_AGE', 'SEX']].copy()
    
    # Convert Transplant Age to Int
    df['TRANSPLANT_AGE'] = df['TRANSPLANT_AGE'].astype(int)
    
    # Filter for only patients on protocol and for results in the first year
    df = df[(df['ON_PROTOCOL'] == 'Y') & (df['PROTOCOL_TESTING_MONTH'].notna())]
    
    # Order dataframe by PID and RESULT_POST_TRANSPLANT
    df = df.sort_values(by=['PID', 'RESULT_POST_TRANSPLANT'])
    
    # Add a Test Count Column
    df['TEST_NUMBER'] = df.groupby(['PID']).cumcount() + 1
    
    # Global logic applied to all risk groups 
    sorted_first_year_df = df.sort_values(['PID', 'TEST_NUMBER']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')
    
    # Moderate Risk
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    
    # Create a column in dataframe to tag mask_mr and mask_lr
    df['RISK_GROUP'] = np.where(mask_mr, 'MR', np.where(mask_lr, 'LR', 'EXCLUDE'))
    
    # Filter out Nan in Risk Group
    df = df[df['RISK_GROUP'] != 'EXCLUDE']
    
    # Create TIME_TO_EVENT Column as well as Binary classification for whether event happened
    df['TIME_TO_EVENT'] = None
    df['EVENT_OCCURRED'] = None
    
    for pid, patient_data in df.groupby('PID'):
        patient_data = patient_data.sort_values(['PID','RESULT_POST_TRANSPLANT'])
        over_cutoff = patient_data[patient_data['RES_RESULT_NUMERIC'] >= 1]
        
        if not over_cutoff.empty:
            # Look for the first time patient receive AS score >=1%
            time_to_event = over_cutoff.iloc[0]['RESULT_POST_TRANSPLANT']
            event_occurred = 1
        else:
            # If not available then take max result post transplant (last test), and event occurred = 0
            time_to_event = patient_data['RESULT_POST_TRANSPLANT'].max()
            event_occurred = 0
        
        df.loc[(df['PID'] == pid), 'TIME_TO_EVENT'] = time_to_event
        df.loc[(df['PID'] == pid), 'EVENT_OCCURRED'] = event_occurred
    
    # Convert to Int
    df['TIME_TO_EVENT'] = df['TIME_TO_EVENT'].astype(int)
    df['EVENT_OCCURRED'] = df['EVENT_OCCURRED'].astype(int)
    
    # Dataframe for Kaplan Meier Curves
    km_df = df[['PID', 'TIME_TO_EVENT', 'EVENT_OCCURRED', 'DONOR_TYPE']].copy()
    km_df = km_df.drop_duplicates()
    km_df = km_df.sort_values(by=['PID', 'TIME_TO_EVENT'])
    
    # Split into two groups
    group_DECEASED = km_df[km_df['DONOR_TYPE'] == 'Deceased']
    group_LIVING = km_df[km_df['DONOR_TYPE'] == 'Living']
    
    # Initialize the KaplanMeierFitter
    kmf_DECEASED = KaplanMeierFitter()
    kmf_LIVING = KaplanMeierFitter()
    
    # Fit models
    kmf_DECEASED.fit(group_DECEASED['TIME_TO_EVENT'], event_observed=group_DECEASED['EVENT_OCCURRED'], label='Deceased')
    kmf_LIVING.fit(group_LIVING['TIME_TO_EVENT'], event_observed=group_LIVING['EVENT_OCCURRED'], label='Living')
    
    # Colors (colorblind-friendly)
    color_deceased = 'rgb(230, 97, 1)'   # orange
    color_living = 'rgb(94, 60, 153)'    # blue-purple
    ci_alpha = 0.18
    
    # Create Plotly figure
    fig = go.Figure()
    # Deceased curve
    fig.add_trace(go.Scatter(
        x=kmf_DECEASED.survival_function_.index,
        y=kmf_DECEASED.survival_function_['Deceased'],
        mode='lines',
        name='Deceased',
        line=dict(color=color_deceased, width=4),
        hoverinfo='x+y+name',
        showlegend=True
    ))
    # Deceased CI
    fig.add_trace(go.Scatter(
        x=list(kmf_DECEASED.confidence_interval_.index) + list(kmf_DECEASED.confidence_interval_.index[::-1]),
        y=list(kmf_DECEASED.confidence_interval_['Deceased_upper_0.95']) + list(kmf_DECEASED.confidence_interval_['Deceased_lower_0.95'][::-1]),
        fill='toself',
        fillcolor=f'rgba(230,97,1,{ci_alpha})',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    # Living curve
    fig.add_trace(go.Scatter(
        x=kmf_LIVING.survival_function_.index,
        y=kmf_LIVING.survival_function_['Living'],
        mode='lines',
        name='Living',
        line=dict(color=color_living, width=4, dash='solid'),
        hoverinfo='x+y+name',
        showlegend=True
    ))
    # Living CI
    fig.add_trace(go.Scatter(
        x=list(kmf_LIVING.confidence_interval_.index) + list(kmf_LIVING.confidence_interval_.index[::-1]),
        y=list(kmf_LIVING.confidence_interval_['Living_upper_0.95']) + list(kmf_LIVING.confidence_interval_['Living_lower_0.95'][::-1]),
        fill='toself',
        fillcolor=f'rgba(94,60,153,{ci_alpha})',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    # Calculate p-value using log-rank test
    from lifelines.statistics import logrank_test
    results = logrank_test(group_DECEASED['TIME_TO_EVENT'], group_LIVING['TIME_TO_EVENT'], 
                          group_DECEASED['EVENT_OCCURRED'], group_LIVING['EVENT_OCCURRED'])
    p_value = results.p_value
    # P-value annotation formatting
    if p_value < 0.0001:
        p_text = f'P-value: {p_value:.2e}'
    else:
        p_text = f'P-value: {p_value:.4f}'
    # Add p-value annotation (large, bold, with white background and border)
    fig.add_annotation(
        x=0.99, y=0.05, xref="paper", yref="paper",
        text=f'<b>{p_text}</b>',
        showarrow=False, font=dict(size=18, color='black', family='Arial'), align="right",
        bgcolor="white", bordercolor="black", borderwidth=2, borderpad=6
    )
    # Update layout for publication style
    fig.update_layout(
        title=dict(text='<b>Survival Curves Between Deceased and Living Donor Groups</b>', x=0.5, xanchor='center', font=dict(size=22)),
        xaxis_title='<b>Time (Days Post-Transplant)</b>',
        yaxis_title='<b>Free from AS>1%</b>',
        xaxis=dict(range=[0, 395], showgrid=False, showline=True, linewidth=2, linecolor='gray', tickfont=dict(size=16), titlefont=dict(size=18)),
        yaxis=dict(showgrid=True, gridcolor='gainsboro', zeroline=False, showline=True, linewidth=2, linecolor='gray', tickfont=dict(size=16), titlefont=dict(size=18)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=540,
        margin=dict(l=100, r=40, t=80, b=60),
        legend=dict(
            x=1.02, y=1, xanchor='left', yanchor='top',
            font=dict(size=16, family='Arial'),
            bordercolor='gray', borderwidth=1, bgcolor='rgba(255,255,255,0.8)'
        ),
        showlegend=True,
    )
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    return fig









