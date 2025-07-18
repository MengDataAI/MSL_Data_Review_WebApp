import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

##################################################################################################
# the input is first_year_df dataframe, and the output is plotly violion for three cohorts 
#
##################################################################################################
def plot_patient_progress_violion(first_year_df):
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

    # Colorblind-friendly palette
    color_map = {
        'Low Risk': 'rgb(94, 60, 153)',      # blue-purple
        'Moderate Risk': 'rgb(230, 97, 1)',  # orange
        'High Risk': 'rgb(60, 180, 75)'      # green
    }
    color_sequence = [color_map['Low Risk'], color_map['Moderate Risk'], color_map['High Risk']]

    # Define the correct order for protocol months
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    
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
        },
        color_discrete_sequence=color_sequence,
        category_orders={"PROTOCOL_TESTING_MONTH": order}
    )

    # Add horizontal reference lines with annotation
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
            text='<b>AlloSure Results Across Low/Moderate/High Risk Groups</b>',
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
        showlegend=True,
        legend=dict(
            title='<b>Risk Group</b>',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=16, family='Arial'),
            bordercolor='gray',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=1200,
        height=540,
        margin=dict(l=100, r=40, t=80, b=60),
        hovermode='closest',
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
        opacity=0.7,
        points=False,
        meanline=dict(color='black', width=3),
        marker=dict(size=8)
    )

    return fig

# test 
plot_patient_progress_violion(first_year_df)
