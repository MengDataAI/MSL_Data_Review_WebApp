import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

##################################################################################################
# the input is first_year_df dataframe, and the output is plotly violion for three cohorts 
#
##################################################################################################
def plot_patient_progress_violion(first_year_df):
    # Data preparation
    sorted_first_year_df = first_year_df.sort_values(['PID', 'RESULT_POST_TRANSPLANT']).copy()
    group_patients = sorted_first_year_df.groupby('PID')
    count = group_patients['RES_RESULT_NUMERIC'].transform('size')
    first_record = group_patients.transform('first')

    # Risk group filters
    mask_hr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] >= 1)
    high_risk_df = sorted_first_year_df[mask_hr].copy()

    mask_mr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 1) & (first_record['RES_RESULT_NUMERIC'] >= 0.5)
    moderate_risk_df = sorted_first_year_df[mask_mr].copy()

    mask_lr = (count >= 2) & (first_record['RES_RESULT_NUMERIC'] < 0.5)
    low_risk_df = sorted_first_year_df[mask_lr].copy()

    # Add risk group labels
    low_risk_df['RISK_DF'] = 'Low Risk'
    moderate_risk_df['RISK_DF'] = 'Moderate Risk'
    high_risk_df['RISK_DF'] = 'High Risk'

    # Combine into one dataframe
    stacked_df = pd.concat([low_risk_df, moderate_risk_df, high_risk_df], ignore_index=True)
    stacked_df = stacked_df[stacked_df['RES_RESULT_NUMERIC'] < 16]

    # Publication-quality colorblind-friendly palette (Nature/Science style)
    color_map = {
        'Low Risk': '#2E8B57',        # Sea Green - represents safety/low risk
        'Moderate Risk': '#FF8C00',   # Dark Orange - represents caution/moderate risk
        'High Risk': '#DC143C'        # Crimson Red - represents danger/high risk
    }

    # Define x-axis order
    protocol_order = ['M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    risk_order = ['Low Risk', 'Moderate Risk', 'High Risk']

    # Order by month, then risk
    x_category_order = [f"{month} - {risk}" for month in protocol_order for risk in risk_order]

    # Create figure
    fig = go.Figure()
    shown_legend = set()

    # MAIN PLOTTING LOOP
    for risk_group in risk_order:
        df_risk = stacked_df[stacked_df['RISK_DF'] == risk_group]
        for protocol_month in protocol_order:
            df_subset = df_risk[df_risk['PROTOCOL_TESTING_MONTH'] == protocol_month]
            if not df_subset.empty:
                x_label = f"{protocol_month} - {risk_group}"
                fig.add_trace(go.Violin(
                    y=df_subset['RES_RESULT_NUMERIC'],
                    x=[x_label] * len(df_subset), 
                    name=risk_group,
                    legendgroup=risk_group,
                    showlegend=risk_group not in shown_legend,
                    line_color=color_map[risk_group],
                    spanmode='hard',
                    span=[0, df_subset['RES_RESULT_NUMERIC'].max()],
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.8,
                    points=False
                ))
                shown_legend.add(risk_group)

    # Add reference lines with correct colors
    fig.add_hline(y=1, line_dash="dash", line_color="darkred", line_width=3,
                  annotation_text="1% threshold", annotation_position="top left",
                  annotation_font_size=14, annotation_font_color="darkred")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", line_width=3,
                  annotation_text="0.5% threshold", annotation_position="top left",
                  annotation_font_size=14, annotation_font_color="orange")

    # Layout styling
    fig.update_layout(
        title=dict(
            text='<b>AlloSure Results Across Low/Moderate/High Risk Groups</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=22, family='Arial, sans-serif')
        ),
        xaxis_title='<b>Protocol Testing Month</b>',
        yaxis_title='<b>AlloSure (% dd-cfDNA)</b>',
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=x_category_order, 
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='gray',
            tickfont=dict(size=14),
            title_font=dict(size=18),
            tickangle=0,
            showticklabels=False
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
            title_font=dict(size=18)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            title=dict(text='<b>Risk Group</b>', font=dict(size=16)),
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            font=dict(size=16),
            bordercolor='gray',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=1200,
        height=540,
        margin=dict(l=100, r=40, t=80, b=60),
        hovermode='closest',
    )

    # Add border box
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='lightgray', width=2), fillcolor='rgba(0,0,0,0)', layer='below')

    return fig

# test 
plot_patient_progress_violion(first_year_df)
