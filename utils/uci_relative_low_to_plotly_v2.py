
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


##################################################################################################
# the input is first_year_df dataframe, and the output is plotly line for low risk > 253% 
#
##################################################################################################
def plot_low_risk_253(first_year_df):
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

    # Add line traces for each patient with publication-quality styling
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, patient_id in enumerate(high_change_patients_low_filtered_df['PID'].unique()):
        patient_data = high_change_patients_low_filtered_df[high_change_patients_low_filtered_df['PID'] == patient_id]
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=patient_data['RESULT_POST_TRANSPLANT'],
            y=patient_data['RES_RESULT_NUMERIC'],
            mode='lines+markers',
            name=f'Patient {patient_id}',
            line=dict(width=2.5, color=color),
            marker=dict(
                symbol='circle',
                size=6,
                color=color,
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            hovertemplate='<b>Patient %{fullData.name}</b><br>' +
                         'Days Post-Transplant: %{x}<br>' +
                         'AlloSure Score: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

    # Add subtle alternating background shading for publication style
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    
    for i, (start, end) in enumerate(intervals):
        color = 'rgba(240, 240, 240, 0.3)' if i % 2 == 0 else 'rgba(255, 255, 255, 0.1)'
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0
        )

    # Add horizontal reference lines with publication-quality styling
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="rgb(214, 39, 40)",
        line_width=3,
        annotation_text="1% Threshold",
        annotation_position="right",
        annotation_font_size=14,
        annotation_font_color="rgb(214, 39, 40)",
        annotation_bgcolor="rgba(255, 255, 255, 0.8)",
        annotation_bordercolor="rgb(214, 39, 40)",
        annotation_borderwidth=1
    )
    
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgb(230, 97, 1)",
        line_width=3,
        annotation_text="0.5% Threshold",
        annotation_position="right",
        annotation_font_size=14,
        annotation_font_color="rgb(230, 97, 1)",
        annotation_bgcolor="rgba(255, 255, 255, 0.8)",
        annotation_bordercolor="rgb(230, 97, 1)",
        annotation_borderwidth=1
    )

    # Update layout for publication-quality styling
    fig.update_layout(
        title=dict(
            text='<b>Low Risk Patients with Sequential AlloSure Score RCV >253%</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='<b>Days Post-Transplant</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            range=[0, 395],
            showgrid=True,
            gridcolor='gainsboro',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gainsboro',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickmode='auto',
            nticks=10,
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            title=dict(text='<b>Patient ID</b>', font=dict(size=14)),
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        width=1000,
        height=600,
        hovermode='closest',
        margin=dict(l=80, r=150, t=80, b=60)
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')

    return fig

# test
plot_low_risk_253(first_year_df)