import numpy as np
import plotly.graph_objects as go
import pandas as pd


##################################################################################################
# the input is first_year_df dataframe, and the output is plotly histogram for relation change value 
#
##################################################################################################
def plot_relative_change_value(first_year_df):
    rcv_df = first_year_df.copy()
    rcv_df['TEST_NUMBER'] = rcv_df.groupby('PID').cumcount() + 1
    rcv_df['RELATIVE_BETWEEN_TEST'] = round(rcv_df.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)
    rcv_plot = rcv_df.loc[rcv_df['RELATIVE_BETWEEN_TEST'].notna()].copy()
    rcv_plot['RCV'] = rcv_plot['RELATIVE_BETWEEN_TEST'].apply(
        lambda x: '≥253' if x >= 253 
        else ('≥149' if x >= 149 
              else ('≥61' if x >= 61 
                    else '<61'))
    )
    # Create summary table
    rcv_summary = rcv_plot.groupby('RCV').agg(
        N=('PID', 'nunique'),
        Median=('RELATIVE_BETWEEN_TEST', 'median'),
        Min=('RELATIVE_BETWEEN_TEST', 'min'),
        Max=('RELATIVE_BETWEEN_TEST', 'max')
    ).reset_index()
    rcv_summary['Median'] = rcv_summary['Median'].round(2)
    rcv_summary['Min'] = rcv_summary['Min'].round(2)
    rcv_summary['Max'] = rcv_summary['Max'].round(2)

    # Create Plotly figure
    fig = go.Figure()
    
    # Add histogram trace with enhanced styling
    fig.add_trace(go.Histogram(
        x=rcv_plot['RELATIVE_BETWEEN_TEST'],
        nbinsx=60,
        marker_color='rgb(94, 60, 153)',  # Blue-purple color
        opacity=0.8,
        marker_line_color='white',
        marker_line_width=1.5,
        name='Frequency'
    ))
    
    # Add background shading for risk categories with distinct light colors
    min_val = rcv_plot['RELATIVE_BETWEEN_TEST'].min()
    max_val = rcv_plot['RELATIVE_BETWEEN_TEST'].max()
    
    # Light blue background (low risk: <61)
    fig.add_vrect(
        x0=min_val, x1=61,
        fillcolor='rgba(173, 216, 230, 0.4)',  # Light blue
        opacity=0.4,
        layer="below",
        line_width=0
    )
    
    # Light yellow background (moderate risk: 61-149)
    fig.add_vrect(
        x0=61, x1=149,
        fillcolor='rgba(255, 255, 224, 0.4)',  # Light yellow
        opacity=0.4,
        layer="below",
        line_width=0
    )
    
    # Light purple background (high risk: >149)
    fig.add_vrect(
        x0=149, x1=max_val,
        fillcolor='rgba(221, 160, 221, 0.4)',  # Light purple
        opacity=0.4,
        layer="below",
        line_width=0
    )
    
    # Add vertical lines at thresholds with annotations
    fig.add_vline(
        x=61,
        line_dash="dash",
        line_color="rgb(60, 180, 75)",
        line_width=3,
        annotation_text="61%",
        annotation_position="top",
        annotation_font_size=16,
        annotation_font_color="rgb(60, 180, 75)",
        annotation_bgcolor="rgba(255, 255, 255, 0.9)",
        annotation_bordercolor="rgb(60, 180, 75)",
        annotation_borderwidth=2
    )
    
    fig.add_vline(
        x=149,
        line_dash="dash",
        line_color="rgb(230, 97, 1)",
        line_width=3,
        annotation_text="149%",
        annotation_position="top",
        annotation_font_size=16,
        annotation_font_color="rgb(230, 97, 1)",
        annotation_bgcolor="rgba(255, 255, 255, 0.9)",
        annotation_bordercolor="rgb(230, 97, 1)",
        annotation_borderwidth=2
    )
    
    # Update layout for publication style
    fig.update_layout(
        title=dict(
            text='<b>Distribution of Relative Change in Between Tests</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=22)
        ),
        xaxis_title='<b>Relative Change Between Sequential Tests (%)</b>',
        yaxis_title='<b>Frequency</b>',
        xaxis=dict(
            range=[min_val, max_val],
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
        width=1000,
        height=540,
        margin=dict(l=100, r=40, t=80, b=60)
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    
    return fig

##################################################################################################
# the input is first_year_df dataframe, and the output is summary table 
#
##################################################################################################
def get_rcv_summary(first_year_df):
    rcv_df = first_year_df.copy()
    rcv_df['TEST_NUMBER'] = rcv_df.groupby('PID').cumcount() + 1
    rcv_df['RELATIVE_BETWEEN_TEST'] = round(rcv_df.groupby('PID')['RES_RESULT_NUMERIC'].pct_change()*100,1)
    rcv_plot = rcv_df.loc[rcv_df['RELATIVE_BETWEEN_TEST'].notna()].copy()
    rcv_plot['RCV'] = rcv_plot['RELATIVE_BETWEEN_TEST'].apply(
        lambda x: '≥253' if x >= 253 
        else ('≥149' if x >= 149 
              else ('≥61' if x >= 61 
                    else '<61'))
    )
    
    # Create summary table
    rcv_summary = rcv_plot.groupby('RCV').agg(
        N=('PID', 'nunique'),
        Median=('RELATIVE_BETWEEN_TEST', 'median'),
        Min=('RELATIVE_BETWEEN_TEST', 'min'),
        Max=('RELATIVE_BETWEEN_TEST', 'max')
    ).reset_index()
    rcv_summary['Median'] = rcv_summary['Median'].round(2)
    rcv_summary['Min'] = rcv_summary['Min'].round(2)
    rcv_summary['Max'] = rcv_summary['Max'].round(2)
    
    return rcv_summary

# test
plot_relative_change_value(first_year_df)
#get_rcv_summary(first_year_df)
