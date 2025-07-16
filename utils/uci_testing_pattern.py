# utils/uci_testing_pattern.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_first_year_df(spark):
    spark_df = spark.read.table("production.data_engineering.UCI_ANALYSIS_PIPELINE")
    df = spark_df.toPandas()
    first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()]
    return first_year_df

def plot_histogram(first_year_df):
    fig, ax = plt.subplots(figsize=(14, 5))
    plt.hist(first_year_df['RESULT_POST_TRANSPLANT'], color='darkblue', alpha=.8, bins=50, edgecolor='white', zorder=2)
    tick_positions = range(0, 400, 14)
    tick_labels = [str(t) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    intervals = [(0,46), (46, 76), (76, 107), (107, 153), (153, 230), (230, 320), (320, 395)]
    for i, (start, end) in enumerate(intervals):
        color = 'grey' if i % 2 == 0 else 'white'
        ax.axvspan(start, end, color=color, alpha=0.35)
    ax.set_xlim(0, 395)
    plt.xlabel('Days Post-Transplant')
    plt.ylabel('Frequency')
    plt.title('AlloSure Testing Frequency')
    plt.tight_layout()
    return fig

def plot_barplot(first_year_df):
    order = [ 'M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    palette = sns.color_palette("deep", len(order))
    grouped_testing_month = first_year_df.groupby('PROTOCOL_TESTING_MONTH').agg({'ACCESSIONID': 'nunique', 'RESULT_POST_TRANSPLANT': 'mean'}).reset_index().sort_values(by='RESULT_POST_TRANSPLANT')
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=grouped_testing_month,
        x="PROTOCOL_TESTING_MONTH",
        y="ACCESSIONID",
        order=order,
        edgecolor='black',
        ax=ax,
        palette=palette
    )
    for i, v in enumerate(grouped_testing_month['ACCESSIONID']):
        ax.text(i, v - 7, str(v), ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')
    ax.set_title('Number of Tests by First Year Schedule')
    ax.set_xlabel("Protocol Testing Month")
    ax.set_ylabel("Number of Tests")
    plt.tight_layout()
    return fig

def plot_violin(first_year_df):
    order = [ 'M1', 'M2', 'M3', 'M4', 'M6', 'M9', 'M12']
    palette = sns.color_palette("deep", len(order))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(
        data=first_year_df,
        x="PROTOCOL_TESTING_MONTH",
        y="RES_RESULT_NUMERIC",
        order=order,
        cut=0,
        ax=ax,
        palette=palette
    )
    plt.axhline(y=1, color='darkred', linestyle='--', linewidth=1)
    ticks = plt.yticks()[0].tolist()
    if 1 not in ticks:
        ticks.append(1)
        ticks = sorted(ticks)
    plt.yticks(ticks)
    ax.set_title('Distribution of AlloSure (% dd-cfDNA) by First Year Schedule')
    ax.set_xlabel("Protocol Testing Month")
    ax.set_ylabel("AlloSure (% dd-cfDNA)")
    plt.tight_layout()
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