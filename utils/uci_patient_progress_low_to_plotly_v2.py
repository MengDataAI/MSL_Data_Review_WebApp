import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

##################################################################################################
# the input is first_year_df dataframe, and the output is violin for low risk patients
#
##################################################################################################
def plot_low_risk_violin(first_year_df):
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()
    month_order = ["M1", "M2", "M3", "M4", "M6", "M9", "M12"]
    
    fig = px.violin(
        low_risk_df,
        x="PROTOCOL_TESTING_MONTH",
        y="RES_RESULT_NUMERIC",
        color="PROTOCOL_TESTING_MONTH",
        title="Low Risk Patients with Initial AlloSure Score <0.5%",
        labels={
            "PROTOCOL_TESTING_MONTH": "Protocol Testing Month",
            "RES_RESULT_NUMERIC": "AlloSure (% dd-cfDNA)"
        },
        category_orders={"PROTOCOL_TESTING_MONTH": month_order},
        color_discrete_sequence=['rgb(94, 60, 153)']  # Blue-purple for low risk
    )
    
    # Add horizontal reference lines with annotations
    fig.add_hline(
        y=1, 
        line_dash="dash", 
        line_color="darkred", 
        line_width=3,
        annotation_text="1% threshold",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="darkred"
    )
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="orange", 
        line_width=3,
        annotation_text="0.5% threshold",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="orange"
    )
    
    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>Low Risk Patients with Initial AlloSure Score <0.5%</b>', 
            x=0.5, 
            xanchor='center',
            font=dict(size=22)
        ),
        xaxis_title='<b>Protocol Testing Month</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18),
            categoryorder='array', 
            categoryarray=month_order
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gainsboro',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        showlegend=False, 
        width=900, 
        height=540,
        margin=dict(l=100, r=40, t=80, b=60),
        hovermode='closest'
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    
    # Update violin plot styling
    fig.update_traces(
        box_visible=True, 
        meanline_visible=True, 
        line_color='black', 
        line_width=2,
        fillcolor='rgba(94, 60, 153, 0.7)',
        opacity=0.8,
        points=False,
        meanline=dict(color='black', width=3),
        marker=dict(size=8)
    )
    return fig

##################################################################################################
# the input is first_year_df dataframe, and the output is line plot for low risk patients
#
##################################################################################################
def plot_low_risk_lines(first_year_df):
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')
    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()
    low_risk_df['TEST_NUMBER'] = low_risk_df.groupby(['PID']).cumcount() + 1
    pid_filter_low = low_risk_df.loc[(low_risk_df['TEST_NUMBER'] != 1) & (low_risk_df['RES_RESULT_NUMERIC'] >= 1), 'PID']
    pid_filter_low = pd.DataFrame(pid_filter_low)
    low_risk_elevated_df = low_risk_df.loc[low_risk_df['PID'].isin(pid_filter_low['PID'])].copy()
    first_test_low = low_risk_elevated_df.loc[low_risk_elevated_df['TEST_NUMBER'] == 1][['PID', 'RESULT_POST_TRANSPLANT']].rename(columns={'RESULT_POST_TRANSPLANT': 'START_DAY'})
    as_100_low = low_risk_elevated_df.loc[low_risk_elevated_df['RES_RESULT_NUMERIC'] >= 1].groupby('PID').first().reset_index()
    result_low = pd.merge(as_100_low[['PID', 'RESULT_POST_TRANSPLANT']], first_test_low, on='PID')
    result_low['TIME_TO_ELEVATED'] = result_low['RESULT_POST_TRANSPLANT'] - result_low['START_DAY']
    mean_time_to_elevated_low = result_low['TIME_TO_ELEVATED'].mean()
    median_time_to_elevated_low = result_low['TIME_TO_ELEVATED'].median()
    ste_time_to_elevated_low = result_low['TIME_TO_ELEVATED'].sem()
    range_min_low = result_low['TIME_TO_ELEVATED'].quantile(0.25)
    range_max_low = result_low['TIME_TO_ELEVATED'].quantile(0.75)
    
    # Colorblind-friendly colors for patient lines (more vibrant)
    colors = [
        'rgb(230, 97, 1)',    # orange
        'rgb(94, 60, 153)',   # blue-purple
        'rgb(60, 180, 75)',   # green
        'rgb(214, 39, 40)',   # red
        'rgb(0, 114, 178)',   # blue
        'rgb(213, 94, 0)',    # dark orange
        'rgb(204, 121, 167)', # pink
        'rgb(230, 159, 0)',   # yellow-orange
        'rgb(86, 180, 233)',  # light blue
        'rgb(0, 158, 115)',   # teal
        'rgb(240, 228, 66)',  # yellow
        'rgb(0, 0, 0)'        # black
    ]
    
    fig = go.Figure()
    
    # Add line traces for each patient
    for i, patient_id in enumerate(low_risk_elevated_df['PID'].unique()):
        patient_data = low_risk_elevated_df[low_risk_elevated_df['PID'] == patient_id]
        fig.add_trace(go.Scatter(
            x=patient_data['RESULT_POST_TRANSPLANT'],
            y=patient_data['RES_RESULT_NUMERIC'],
            mode='lines+markers',
            name=f'Patient {patient_id}',
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(symbol='circle', size=8, opacity=0.8, color=colors[i % len(colors)]),
            hovertemplate='<b>Patient %{fullData.name}</b><br>' +
                         'Days Post-Transplant: %{x}<br>' +
                         'AlloSure Score: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Add alternating background shading for intervals
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    for i, (start, end) in enumerate(intervals):
        color = 'lightgray' if i % 2 == 0 else 'white'
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.35, layer="below", line_width=0)
    
    # Add horizontal reference lines with annotations
    fig.add_hline(
        y=1, 
        line_dash="dash", 
        line_color="darkred", 
        line_width=3,
        annotation_text="1% threshold",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="darkred"
    )
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="orange", 
        line_width=3,
        annotation_text="0.5% threshold",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="orange"
    )
    
    # Add summary statistics as annotations with better styling
    fig.add_annotation(
        x=0.99, y=0.95, xref="paper", yref="paper",
        text=f"<b>Average Time to Elevated Result (STE): {mean_time_to_elevated_low:.0f} ({ste_time_to_elevated_low:.0f}) days</b>",
        showarrow=False, 
        font=dict(size=14, color='black', family='Arial'), 
        align="right", 
        bgcolor="white", 
        bordercolor="black", 
        borderwidth=2,
        borderpad=6
    )
    fig.add_annotation(
        x=0.99, y=0.88, xref="paper", yref="paper",
        text=f"<b>Median (IQR): {median_time_to_elevated_low:.0f} ({range_min_low:.0f} - {range_max_low:.0f}) days</b>",
        showarrow=False, 
        font=dict(size=14, color='black', family='Arial'), 
        align="right", 
        bgcolor="white", 
        bordercolor="black", 
        borderwidth=2,
        borderpad=6
    )
    
    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>Low Risk Patients with Subsequent AlloSure Score ≥ 1%</b>', 
            x=0.5, 
            xanchor='center',
            font=dict(size=22)
        ),
        xaxis_title='<b>Days Post-Transplant</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            range=[0, 395], 
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gainsboro',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        showlegend=True,
        legend=dict(
            title='<b>Patient ID</b>', 
            x=1.02, 
            y=1, 
            xanchor='left', 
            yanchor='top',
            font=dict(size=14, family='Arial'),
            bordercolor='gray',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=1000, 
        height=540, 
        margin=dict(l=100, r=40, t=80, b=60),
        hovermode='closest'
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    
    return fig

##################################################################################################
# the input is first_year_df dataframe, and the output is statistics table for violin
#
##################################################################################################
def get_low_risk_summary_table(first_year_df):
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


# test 
#plot_low_risk_violin(first_year_df)
#get_low_risk_summary_table(first_year_df)
plot_low_risk_lines(first_year_df)
