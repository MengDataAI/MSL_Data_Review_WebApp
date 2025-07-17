
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


##################################################################################################
# the input is first_year_df dataframe, and the output is plotly violion for high risk
#
##################################################################################################
def plot_high_risk_progress_violin(first_year_df):
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
        category_orders={"PROTOCOL_TESTING_MONTH": ["M1", "M2", "M3", "M4", "M6", "M9", "M12"]},
        color_discrete_sequence=['rgb(60, 180, 75)']  # Green for high risk
    )

    # Add horizontal reference line with annotation
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

    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>High Risk Patients with Initial AlloSure Score ≥ 1%</b>',
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
        fillcolor='rgba(60, 180, 75, 0.7)',
        opacity=0.8,
        points=False,
        meanline=dict(color='black', width=3),
        marker=dict(size=8)
    )

    return fig

##################################################################################################
# the input is first_year_df dataframe, and the output is summary table for violion
#
##################################################################################################
def get_high_risk_summary_table(first_year_df):
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

##################################################################################################
# the input is first_year_df dataframe, and the output is line plot for high risk patients
#
##################################################################################################
def plot_high_risk_lines(first_year_df):
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

    # Add line traces for each patient
    for i, patient_id in enumerate(high_risk_elevated_df['PID'].unique()):
        patient_data = high_risk_elevated_df[high_risk_elevated_df['PID'] == patient_id]
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
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=0.35,
            layer="below",
            line_width=0
        )

    # Add horizontal reference line with annotation
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

    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>High Risk Patients with Subsequent AlloSure Score ≥ 1%</b>',
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


# test 
plot_high_risk_progress_violin(first_year_df)
#get_high_risk_summary_table(first_year_df)
#plot_high_risk_lines(first_year_df)