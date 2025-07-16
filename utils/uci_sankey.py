import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import plotly.graph_objects as go

def generate_uci_sankey(df: pd.DataFrame):
    """
    Generate a Sankey plotly figure for UCI patient journey.
    Args:
        df (pd.DataFrame): DataFrame with required columns.
    Returns:
        fig (plotly.graph_objs.Figure): Sankey diagram figure.
    """
    # Filter dataset on valid first year months 
    first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()].copy()

    # Create a copy of the first_record_df
    sankey_df = first_year_df.copy()

    # Add Test Number for each Patient
    sankey_df['TEST_NUMBER'] = sankey_df.groupby('PID').cumcount() + 1

    # Create field to assign unique value for source/target
    def sankey_code(row):
        month = row['PROTOCOL_TESTING_MONTH']
        risk = row['RISK_LEVEL']
        mapping = {
            ('M1', 'HR'): 0, ('M1', 'MR'): 1, ('M1', 'LR'): 2,
            ('M2', 'HR'): 3, ('M2', 'MR'): 4, ('M2', 'LR'): 5,
            ('M3', 'HR'): 6, ('M3', 'MR'): 7, ('M3', 'LR'): 8,
            ('M4', 'HR'): 9, ('M4', 'MR'): 10, ('M4', 'LR'): 11,
            ('M6', 'HR'): 12, ('M6', 'MR'): 13, ('M6', 'LR'): 14,
            ('M9', 'HR'): 15, ('M9', 'MR'): 16, ('M9', 'LR'): 17,
            ('M12', 'HR'): 18, ('M12', 'MR'): 19, ('M12', 'LR'): 20,
        }
        return mapping.get((month, risk), None)
    sankey_df['SANKEY'] = sankey_df.apply(sankey_code, axis=1)

    # Group by PID, PROTOCOL_TESTING_MONTH, RISK_LEVEL, and count unique accession
    sankey_df = sankey_df.groupby(['PID', 'PROTOCOL_TESTING_MONTH', 'RISK_LEVEL', 'SANKEY']).agg({'ACCESSIONID': 'nunique'}).reset_index()
    sankey_df = sankey_df.sort_values(by=['PID', 'SANKEY'], ascending=[True, True])

    # Create source and target columns
    sankey_df['SOURCE'] = sankey_df['SANKEY']
    sankey_df['TARGET'] = sankey_df.groupby('PID')['SANKEY'].shift(-1)

    # Drop rows where TARGET is NaN (last row in each patient)
    df_links = sankey_df.dropna(subset=['TARGET']).copy()
    df_links['TARGET'] = df_links['TARGET'].astype(int)
    df_links = df_links[['PID', 'SOURCE', 'TARGET', 'ACCESSIONID']]

    # Helper to convert hex to rgba with alpha
    def hex_to_rgba(hex_color, alpha):
        rgba = mcolors.to_rgba(hex_color, alpha)
        return f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {rgba[2]*255}, {rgba[3]})'

    # Define node labels and colors
    labels = ['HR M1', 'MR M1', 'LR M1', 
              'HR M2', 'MR M2', 'LR M2', 
              'HR M3', 'MR M3', 'LR M3', 
              'HR M4', 'MR M4', 'LR M4', 
              'HR M6', 'MR M6', 'LR M6', 
              'HR M9', 'MR M9', 'LR M9', 
              'HR M12', 'MR M12', 'LR M12']
    node_colors = ['#d62728', '#ff7f0e', '#2ca02c'] * 7 

    # Map source index to node color and lighten it for links
    source = df_links['SOURCE'].values
    target = df_links['TARGET'].values
    value = df_links['ACCESSIONID'].values
    link_alpha = 0.5
    link_colors = [hex_to_rgba(node_colors[s], link_alpha) for s in source]

    # Create Sankey figure
    fig = go.Figure(go.Sankey(
        node = dict(
            pad = 50,
            thickness = 15,
            line = dict(color = "black", width = 0.3),
            label = labels,
            color = node_colors,
            x = [0.1, 0.1, 0.1, 
                 0.2, 0.2, 0.2,
                 0.3, 0.3, 0.3,
                 0.4, 0.4, 0.4,
                 0.6, 0.6, 0.6,
                 0.8, 0.8, 0.8,
                 1.0, 1.0, 1.0],
            y = [0.1, 0.25, 1,
                 0.1, 0.25, 1,
                 0.1, 0.25, 1,
                 0.1, 0.25, 1,
                 0.1, 0.25, 1,
                 0.1, 0.25, 1,
                 0.1, 0.25, 1],
        ),
        link = dict(
            source = source,
            target = target,
            value = value,
            color = link_colors
        )
    ))

    fig.update_layout(
        title_text="First Year AlloSure Testing Flow",  
        font_size=15,
        width=1300,
        height=800,
        margin=dict(t=40, b=270, l=10, r=10)
    )
    return fig




