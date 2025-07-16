# utils/uci_relative_change_value.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    # Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(data=rcv_plot, x='RELATIVE_BETWEEN_TEST', bins=60, color='black', edgecolor='white', linewidth=.8, zorder=2, alpha=.8)
    plt.axvspan(rcv_plot['RELATIVE_BETWEEN_TEST'].min(), 61, color='green', alpha=0.2)
    plt.axvspan(61, 149, color='yellow', alpha=0.2)
    plt.axvspan(149, rcv_plot['RELATIVE_BETWEEN_TEST'].max(), color='red', alpha=0.1)
    plt.axvline(x=61, color='green', linestyle='--', linewidth=1)
    plt.axvline(x=149, color='yellow', linestyle='--', linewidth=1)
    plt.xlim(rcv_plot['RELATIVE_BETWEEN_TEST'].min(), rcv_plot['RELATIVE_BETWEEN_TEST'].max())
    plt.title('Distribution of Relative Change in Between Tests')
    plt.xlabel('Relative Change Between Sequential Tests (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    fig = plt.gcf()
    return fig, rcv_summary










