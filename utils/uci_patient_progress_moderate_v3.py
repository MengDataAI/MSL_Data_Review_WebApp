import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

##################################################################################################
# the input is first_year_df dataframe, and the output is violin plot for moderate risk patients
#
##################################################################################################
def plot_moderate_risk_violin(first_year_df):
    # Data preparation
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Filter for MR (initial result between 0.5% <= result < 1% ) with atleast 2 tests results
    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    # Define protocol order
    protocol_order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']

    # Create figure
    fig = go.Figure()

    # Plot violin for each protocol month
    for protocol_month in protocol_order:
        df_subset = moderate_risk_df[moderate_risk_df['PROTOCOL_TESTING_MONTH'] == protocol_month]
        if not df_subset.empty:
            fig.add_trace(go.Violin(
                y=df_subset['RES_RESULT_NUMERIC'],
                x=[protocol_month] * len(df_subset),
                name='Moderate Risk',
                line_color='#FF8C00',  # Dark Orange for moderate risk (Nature/Science standard)
                spanmode='hard',
                span=[0, df_subset['RES_RESULT_NUMERIC'].max()],
                box_visible=True,
                meanline_visible=True,
                opacity=0.8,
                points=False,
                showlegend=False
            ))

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
            text='<b>Moderate Risk Patients with Initial AlloSure Score ≥ 0.5 - < 1%</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=22)
        ),
        xaxis_title='<b>Protocol Testing Month</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=protocol_order,
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            range=[0, None],  # Start from 0, no upper limit
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

    return fig

##################################################################################################
# the input is first_year_df dataframe, and the output is line plot for moderate risk patients
#
##################################################################################################
def plot_moderate_risk_lines(first_year_df):
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
    for i, patient_id in enumerate(moderate_risk_elevated_df['PID'].unique()):
        patient_data = moderate_risk_elevated_df[moderate_risk_elevated_df['PID'] == patient_id]
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

    # Add horizontal reference lines with right-aligned annotations
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="darkred",
        line_width=3,
        annotation_text="1% threshold",
        annotation_position="top right",
        annotation_font_size=14,
        annotation_font_color="darkred",
        annotation_bgcolor="rgba(255, 255, 255, 0.8)",
        annotation_bordercolor="darkred",
        annotation_borderwidth=1
    )
    
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="orange",
        line_width=3,
        annotation_text="0.5% threshold",
        annotation_position="top right",
        annotation_font_size=14,
        annotation_font_color="orange",
        annotation_bgcolor="rgba(255, 255, 255, 0.8)",
        annotation_bordercolor="orange",
        annotation_borderwidth=1
    )

    # Add summary statistics as annotations with better styling (centered at top)
    fig.add_annotation(
        x=0.5, y=0.99,
        xref="paper", yref="paper",
        text=f"<b>Average Time to Elevated Result (STE): {mean_time_to_elevated_moderate:.0f} ({ste_time_to_elevated_moderate:.0f}) days</b>",
        showarrow=False,
        font=dict(size=14, color='black', family='Arial'),
        align="center",
        bgcolor="white",
        borderwidth=0,
        borderpad=6
    )
    
    fig.add_annotation(
        x=0.5, y=0.88,
        xref="paper", yref="paper",
        text=f"<b>Median (IQR): {median_time_to_elevated_moderate:.0f} ({range_min_moderate:.0f} - {range_max_moderate:.0f}) days</b>",
        showarrow=False,
        font=dict(size=14, color='black', family='Arial'),
        align="center",
        bgcolor="white",
        borderwidth=0,
        borderpad=6
    )

    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>Moderate Risk Patients with Subsequent AlloSure Score ≥ 1%</b>',
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
            range=[0, 3],
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
            y=0.95,
            xanchor='left',
            yanchor='top',
            font=dict(size=12, family='Arial'),
            bordercolor='gray',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.8)',
            itemsizing='constant'
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
# the input is first_year_df dataframe, and the output is summary table for moderate risk patients
#
##################################################################################################
def get_moderate_risk_summary_table(first_year_df):
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

# test 
plot_moderate_risk_violin(first_year_df)
#get_moderate_risk_summary_table(first_year_df)
#plot_moderate_risk_lines(first_year_df)