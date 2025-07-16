#!/usr/bin/env python3
"""
MSL Data Review Web Application - Streamlit Prototype
This application allows physicians to upload patient IDs and generate plots from data review.
07-15-2025
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import logging

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_fetcher import MSLDataFetcher
#from plot_generator import PlotGenerator
from config.settings import load_config
from utils.uci_sankey import generate_uci_sankey
from utils.uci_testing_pattern import (
    get_first_year_df, plot_histogram, plot_barplot, plot_violin, get_statistics_table
)
from utils.uci_relative_change_value_to_plotly import plot_relative_change_value
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MSL Data Review",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better layout
st.markdown("""
<style>
    .main-header {
        background-color: #007bff;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .control-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    /* Style buttons to look like text - remove borders and left align */
    .stButton > button {
        background: none !important;
        border: none !important;
        color: #333;
        text-align: left !important;
        padding: 0.5rem 0;
        font-weight: normal;
        box-shadow: none !important;
        border-radius: 0 !important;
        width: 100%;
        margin: 0;
        justify-content: flex-start !important;
        display: flex !important;
    }
    .stButton > button:hover {
        background: #f0f0f0 !important;
        color: #007bff;
        border: none !important;
        box-shadow: none !important;
    }
    .stButton > button:focus {
        border: none !important;
        box-shadow: none !important;
    }
    /* Style for section headers */
    .analysis-section-header {
        font-weight: bold;
        color: #007bff;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    /* Ensure button content is left-aligned */
    .stButton > button > div {
        text-align: left !important;
        width: 100%;
    }
    /* Hierarchical indentation for sub-functions */
    .stButton > button[key="btn_moderate_253"],
    .stButton > button[key="btn_low_253"],
    .stButton > button[key="btn_high_risk"],
    .stButton > button[key="btn_moderate_risk"],
    .stButton > button[key="btn_low_risk"] {
        margin-left: 2rem !important;
        border-left: 3px solid #007bff !important;
        padding-left: 1rem !important;
    }
    .load-data-btn button {
        background: linear-gradient(90deg, #ff3b3b 0%, #ff7b7b 100%) !important; /* Red gradient */
        color: #fff !important;
        font-weight: bold !important;
        font-size: 1.3rem !important;   /* Bigger font */
        border: none !important;
        border-radius: 8px !important;
        padding: 0.9rem 0 !important;   /* Slightly more padding for bigger button */
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px rgba(255,59,59,0.15) !important;
        transition: background 0.3s;
        letter-spacing: 0.5px;
    }
    .load-data-btn button:hover {
        background: linear-gradient(90deg, #b30000 0%, #ff3b3b 100%) !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🏥 MSL Data Review Assistant</h1></div>', unsafe_allow_html=True)
    
    # Create two columns: left panel (1/3) and right main area (2/3)
    left_panel, right_main = st.columns([1, 2])
    
    with left_panel:
        # Analysis Window Controls
        st.markdown('<div class="control-section"><h4>Analysis Window</h4></div>', unsafe_allow_html=True)
        time_start = st.date_input("Analysis Window Startpoint", value=datetime.now() - timedelta(days=30))
        time_end = st.date_input("Analysis Window Endpoint", value=datetime.now())
        
        # Patient IDs Input
        st.markdown('<div class="control-section"><h4> Patient IDs</h4></div>', unsafe_allow_html=True)
        patient_ids_input = st.text_area(
            "Input Patient IDs",
            height=100,
            placeholder="Enter patient IDs (one per line or comma-separated)\nExample:\nPAT001\nPAT002\nPAT003"
        )
        
        # Load Data Button
        with st.container():
            load_data_clicked = st.button("Load Data", key="load_data_btn", use_container_width=True)
            st.markdown(
                """
                <style>
                div[data-testid="stButton"][key="load_data_btn"] > button {
                    background: linear-gradient(90deg, #ff3b3b 0%, #ff7b7b 100%) !important;
                    color: #fff !important;
                    font-weight: 900 !important;
                    font-size: 1.5rem !important;
                    border: none !important;
                    border-radius: 12px !important;
                    padding: 1.1rem 0 !important;
                    margin-top: 1rem !important;
                    margin-bottom: 1.5rem !important;
                    box-shadow: 0 4px 16px rgba(255,59,59,0.18) !important;
                    transition: background 0.3s, box-shadow 0.3s;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                    outline: none !important;
                }
                div[data-testid="stButton"][key="load_data_btn"] > button:hover {
                    background: linear-gradient(90deg, #b30000 0%, #ff3b3b 100%) !important;
                    color: #fff !important;
                    box-shadow: 0 6px 24px rgba(255,59,59,0.28) !important;
                    outline: none !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        # Use session state to store the result so it persists after rerun
        if 'query_result_df' not in st.session_state:
            st.session_state['query_result_df'] = None
        if 'query_error' not in st.session_state:
            st.session_state['query_error'] = None

        if load_data_clicked:
            # Parse patient IDs
            patient_ids = fetcher.parse_patient_ids(patient_ids_input)
            # Format dates
            start_date = time_start.strftime("%Y-%m-%d")
            end_date = time_end.strftime("%Y-%m-%d")
            # Fetch data
            with st.spinner("Loading data..."):
                try:
                    df = fetcher.fetch_custom_patient_data(patient_ids, start_date, end_date)
                    if not df.empty:
                        st.success(f"Loaded {len(df)} records.")
                        st.dataframe(df)
                        st.session_state['uci_df'] = df  # Store for later use
                    else:
                        st.warning("No data found for the given criteria.")
                        st.session_state['uci_df'] = None
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    st.session_state['uci_df'] = None

    # --- Restore Analysis Options section ---
    st.subheader("Analysis Options")

    # Main and sub-function buttons (styled as text)
    show_patient_journey = st.button("1. Patient Journey Sankey", key="btn_patient_journey", use_container_width=True)
    show_testing_pattern = st.button("2. Testing Pattern", key="btn_testing_pattern", use_container_width=True)
    show_relative_change = st.button("3. Relative Change Value", key="btn_relative_change", use_container_width=True)
    show_moderate_253 = st.button("Relative Change Value: Moderate Risk >253%", key="btn_moderate_253", use_container_width=True)
    show_low_253 = st.button("Relative Change Value: Low Risk >253%", key="btn_low_253", use_container_width=True)
    show_patient_progress = st.button("4. Patient Progress", key="btn_patient_progress", use_container_width=True)
    show_high_risk = st.button("Patient Progress: High Risk", key="btn_high_risk", use_container_width=True)
    show_moderate_risk = st.button("Patient Progress: Moderate Risk", key="btn_moderate_risk", use_container_width=True)
    show_low_risk = st.button("Patient Progress: Low Risk", key="btn_low_risk", use_container_width=True)

    with right_main:
        st.markdown('<div class="plot-container"><h3> Analysis Results</h3></div>', unsafe_allow_html=True)
        if st.session_state.get('query_error'):
            st.error(f"Error loading data: {st.session_state['query_error']}")
        elif st.session_state.get('query_result_df') is not None:
            df = st.session_state['query_result_df']
            if not df.empty:
                st.success(f"Loaded {len(df)} records.")
                st.dataframe(df)
            else:
                st.warning("No data found for the given criteria.")
        else:
            st.info("Use the left panel to configure your analysis and generate plots")
            #pass
        
        # Display placeholder for plots
        #st.info("Use the left panel to configure your analysis and generate plots")
        
        # Example of how plots will be displayed
        if show_patient_journey:
            df = st.session_state.get('uci_df')
            if df is not None and not df.empty:
                fig = generate_uci_sankey(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please load data first before generating the Sankey plot.")
        
        if show_testing_pattern:
            df = st.session_state.get('uci_df')
            if df is not None and not df.empty:
                # Use the same filtering as in your Sankey code
                first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()].copy()
                with st.spinner("Generating testing pattern plots..."):
                    fig1 = plot_histogram(first_year_df)
                    fig2 = plot_barplot(first_year_df)
                    fig3 = plot_violin(first_year_df)
                    stats_table = get_statistics_table(first_year_df)

                    st.pyplot(fig1)
                    st.pyplot(fig2)
                    st.pyplot(fig3)
                    st.markdown("### AlloSure Results Summary Table")
                    st.dataframe(stats_table)
            else:
                st.warning("Please load data first before generating the Testing Pattern plots.")
        
        if show_relative_change:
            df = st.session_state.get('uci_df')
            if df is not None and not df.empty:
                # Use the same filtering as in your Sankey code
                first_year_df = df[df['PROTOCOL_TESTING_MONTH'].notna()].copy()
                with st.spinner("Generating relative change value plot..."):
                    fig, summary_table = plot_relative_change_value(first_year_df)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### Relative Change Value Summary Table")
                    st.dataframe(summary_table)
            else:
                st.warning("Please load data first before generating the Relative Change Value plot.")
        
        # if show_moderate_risk_253:
        #     st.markdown('<div class="plot-container"><h4> Moderate Risk >253%</h4></div>', unsafe_allow_html=True)
        #     st.info("Moderate Risk >253% plot will be displayed here")
        
        # if show_low_risk_253:
        #     st.markdown('<div class="plot-container"><h4> Low Risk >253%</h4></div>', unsafe_allow_html=True)
        #     st.info("Low Risk >253% plot will be displayed here")
        
        # if show_patient_progress:
        #     st.markdown('<div class="plot-container"><h4> Patient Progress</h4></div>', unsafe_allow_html=True)
        #     st.info("Patient Progress plot will be displayed here")
        
        # if show_high_risk:
        #     st.markdown('<div class="plot-container"><h4> High Risk Analysis</h4></div>', unsafe_allow_html=True)
        #     st.info("High Risk Analysis plot will be displayed here")
        
        # if show_moderate_risk:
        #     st.markdown('<div class="plot-container"><h4>🟡 Moderate Risk Analysis</h4></div>', unsafe_allow_html=True)
        #     st.info("Moderate Risk Analysis plot will be displayed here")
        
        # if show_low_risk:
        #     st.markdown('<div class="plot-container"><h4>🟢 Low Risk Analysis</h4></div>', unsafe_allow_html=True)
        #     st.info("Low Risk Analysis plot will be displayed here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MSL Data Analysis Dashboard** | "
        "Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    config = load_config()
    fetcher = MSLDataFetcher(config)
    main() 