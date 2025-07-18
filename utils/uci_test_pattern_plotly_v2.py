import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

##################################################################################################
# the input is first_year_df dataframe, and the output is plotly histogram
#
##################################################################################################
def plot_histogram(first_year_df):
    # Create the histogram
    fig = go.Figure()
    
    # Add histogram trace with publication-quality styling
    fig.add_trace(go.Histogram(
        x=first_year_df['RESULT_POST_TRANSPLANT'],
        nbinsx=50,
        marker_color='rgb(94, 60, 153)',  # Blue-purple color
        opacity=0.8,
        marker_line_color='white',
        marker_line_width=1.5,
        name='Frequency'
    ))
    
    # Add alternating background shading for protocol months
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    colors = ['grey', 'white', 'grey', 'white', 'grey', 'white', 'grey']
    
    for i, ((start, end), color) in enumerate(zip(intervals, colors)):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=0.35,
            layer="below",
            line_width=0
        )
    
    # Update layout for publication-quality styling
    fig.update_layout(
        title=dict(
            text='<b>AlloSure Testing Frequency</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='<b>Days Post-Transplant</b>',
        yaxis_title='<b>Frequency</b>',
        xaxis=dict(
            range=[0, 395],
            dtick=14,
            tickmode='linear',
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        width=1000,
        height=500,
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    
    return fig


##################################################################################################
# the input is first_year_df dataframe, and the output is plotly barplot
#
##################################################################################################
def plot_barplot(first_year_df):
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    # Professional color palette for publication
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Group and sort data
    grouped_testing_month = first_year_df.groupby('PROTOCOL_TESTING_MONTH').agg({
        'ACCESSIONID': 'nunique', 
        'RESULT_POST_TRANSPLANT': 'mean'
    }).reset_index()
    
    # Ensure proper ordering
    grouped_testing_month['PROTOCOL_TESTING_MONTH'] = pd.Categorical(
        grouped_testing_month['PROTOCOL_TESTING_MONTH'], 
        categories=order, 
        ordered=True
    )
    grouped_testing_month = grouped_testing_month.sort_values('PROTOCOL_TESTING_MONTH')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=grouped_testing_month['PROTOCOL_TESTING_MONTH'],
        y=grouped_testing_month['ACCESSIONID'],
        marker_color=palette,
        marker_line_color='white',
        marker_line_width=2,
        text=grouped_testing_month['ACCESSIONID'],
        textposition='outside',
        textfont=dict(color='black', size=14, family='Arial Black'),
        name='Number of Tests'
    ))
    
    # Update layout for publication-quality styling
    fig.update_layout(
        title=dict(
            text='<b>Number of Tests by First Year Schedule</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='<b>Protocol Testing Month</b>',
        yaxis_title='<b>Number of Tests</b>',
        xaxis=dict(
            showgrid=False,
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
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        width=800,
        height=500,
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    
    return fig


##################################################################################################
# the input is first_year_df dataframe, and the output is plotly violin
#
##################################################################################################
def plot_violin(first_year_df):
    order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    # Professional color palette for publication
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Create violin plot
    fig = go.Figure()
    
    for i, month in enumerate(order):
        month_data = first_year_df[first_year_df['PROTOCOL_TESTING_MONTH'] == month]['RES_RESULT_NUMERIC']
        if not month_data.empty:
            fig.add_trace(go.Violin(
                x=[month] * len(month_data),
                y=month_data,
                name=month,
                line_color=palette[i],
                fillcolor=palette[i],
                opacity=0.6,
                line_width=2,
                showlegend=False
            ))
    
    # Add horizontal reference line with publication-quality styling
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
    
    # Update layout for publication-quality styling
    fig.update_layout(
        title=dict(
            text='<b>Distribution of AlloSure (% dd-cfDNA) by First Year Schedule</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='<b>Protocol Testing Month</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            categoryorder='array', 
            categoryarray=order,
            showgrid=False,
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
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        width=1000,
        height=500,
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    # Add a subtle border to the plot area
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')
    
    return fig

def get_statistics_table(first_year_df):
    summary_stats_2b = first_year_df.groupby('PROTOCOL_TESTING_MONTH').agg(
        N = ('ACCESSIONID', 'nunique'),
        Median = ("RES_RESULT_NUMERIC", "median"),
        Min = ("RES_RESULT_NUMERIC", "min"),
        Max = ("RES_RESULT_NUMERIC", "max"),
        p_25 = ("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.25)),
        p_75 = ("RES_RESULT_NUMERIC", lambda x: np.quantile(x, 0.75))).reset_index()
    summary_stats_2b['p_25'] = summary_stats_2b['p_25'].round(2)
    summary_stats_2b['p_75'] = summary_stats_2b['p_75'].round(2)
    summary_stats_2b['Min'] = summary_stats_2b['Min'].round(2)
    summary_stats_2b['Max'] = summary_stats_2b['Max'].round(2)
    summary_stats_2b['Median'] = summary_stats_2b['Median'].round(2)
    summary_stats_2b['Median (IQR)'] = (summary_stats_2b['Median'].astype(str) +  " (" + summary_stats_2b['p_25'].astype(str) + ", " + summary_stats_2b['p_75'].astype(str) + ")")
    summary_stats_2b['Range'] = ( summary_stats_2b['Min'].astype(str) + ", " + summary_stats_2b['Max'].astype(str))
    summary_stats_2b.drop(['Min', 'Max', 'Median', 'p_25', 'p_75'], axis=1, inplace=True)
    order = [ 'M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    first_col = summary_stats_2b.columns[0]
    summary_stats_2b[first_col] = pd.Categorical(summary_stats_2b[first_col], categories=order, ordered=True)
    summary_stats_2b.sort_values(first_col, inplace=True, ignore_index=True)
    rotated = (summary_stats_2b.set_index('PROTOCOL_TESTING_MONTH').T.reset_index()).rename(columns={'index': 'Protocol Testing Month'})
    return rotated

# test
plot_violin(first_year_df)
#plot_barplot(first_year_df)
#plot_histogram(first_year_df)
#get_statistics_table(first_year_df)