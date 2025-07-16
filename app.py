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
from plot_generator import PlotGenerator
from config.settings import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MSL Data Review",
    page_icon="🏥",
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
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🏥 MSL Data Review Assistant</h1></div>', unsafe_allow_html=True)
    
    # Create two columns: left panel (1/3) and right main area (2/3)
    left_panel, right_main = st.columns([1, 2])
    
    with left_panel:
        st.markdown('<div class="control-section"><h3>📋 Control Panel</h3></div>', unsafe_allow_html=True)
        
        # Time Window Controls
        st.markdown('<div class="control-section"><h4>⏰ Time Window</h4></div>', unsafe_allow_html=True)
        time_start = st.date_input("Time Window Startpoint", value=datetime.now() - timedelta(days=30))
        time_end = st.date_input("Time Window Endpoint", value=datetime.now())
        
        # Patient IDs Input
        st.markdown('<div class="control-section"><h4>👤 Patient IDs</h4></div>', unsafe_allow_html=True)
        patient_ids_input = st.text_area(
            "Input Box for Patient IDs",
            height=100,
            placeholder="Enter patient IDs (one per line or comma-separated)\nExample:\nPAT001\nPAT002\nPAT003"
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Or Upload Patient IDs File",
            type=['csv', 'txt', 'xlsx'],
            help="Upload a file containing patient IDs"
        )
        
        # Load Data Button
        if st.button("🔄 Load Data", type="primary", use_container_width=True):
            st.info("Data loading functionality will be implemented here")
        
        # Analysis Options
        st.markdown('<div class="control-section"><h4>📊 Analysis Options</h4></div>', unsafe_allow_html=True)
        
        # Checkboxes for different plots
        show_patient_journey = st.checkbox("Patient Journey Sankey")
        show_testing_pattern = st.checkbox("Testing Pattern")
        show_patient_progress = st.checkbox("Patient Progress")
        show_high_risk = st.checkbox("High Risk")
        show_moderate_risk = st.checkbox("Moderate Risk")
        show_low_risk = st.checkbox("Low Risk")
        
        # Risk thresholds
        st.markdown('<div class="control-section"><h4>⚖️ Risk Thresholds</h4></div>', unsafe_allow_html=True)
        relative_change_value = st.number_input("Relative Change Value", value=0.0, step=0.1)
        moderate_risk_threshold = st.number_input("Moderate Risk >253%", value=253.0, step=0.1)
        low_risk_threshold = st.number_input("Low Risk >253%", value=253.0, step=0.1)
        
        # Generate Plots Button
        if st.button("📈 Generate Plots", type="secondary", use_container_width=True):
            st.info("Plot generation functionality will be implemented here")
    
    with right_main:
        st.markdown('<div class="plot-container"><h3>📊 Analysis Results</h3></div>', unsafe_allow_html=True)
        
        # Display placeholder for plots
        st.info("Use the left panel to configure your analysis and generate plots")
        
        # Example of how plots will be displayed
        if show_patient_journey:
            st.markdown('<div class="plot-container"><h4>🔄 Patient Journey Sankey</h4></div>', unsafe_allow_html=True)
            st.info("Patient Journey Sankey plot will be displayed here")
        
        if show_testing_pattern:
            st.markdown('<div class="plot-container"><h4>📈 Testing Pattern</h4></div>', unsafe_allow_html=True)
            st.info("Testing Pattern plot will be displayed here")
        
        if show_patient_progress:
            st.markdown('<div class="plot-container"><h4>�� Patient Progress</h4></div>', unsafe_allow_html=True)
            st.info("Patient Progress plot will be displayed here")
        
        if show_high_risk:
            st.markdown('<div class="plot-container"><h4>🔴 High Risk Analysis</h4></div>', unsafe_allow_html=True)
            st.info("High Risk Analysis plot will be displayed here")
        
        if show_moderate_risk:
            st.markdown('<div class="plot-container"><h4>🟡 Moderate Risk Analysis</h4></div>', unsafe_allow_html=True)
            st.info("Moderate Risk Analysis plot will be displayed here")
        
        if show_low_risk:
            st.markdown('<div class="plot-container"><h4>🟢 Low Risk Analysis</h4></div>', unsafe_allow_html=True)
            st.info("Low Risk Analysis plot will be displayed here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MSL Data Analysis Dashboard** | "
        "Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main() 