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
from datetime import datetime
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

def main():
    """Main application function"""
    
    # Load configuration
    config = load_config()
    
    # Header
    st.title("🏥 MSL Data Review Assistant")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            ["data_engineering.uci_analysis_pipeline", "Other Database"],
            help="Select the database table to query"
        )
        
        # Analysis type selection
        analysis_type = st.multiselect(
            "Analysis Types",
            ["Patient Demographics", "Lab Results", "Treatment History", "Outcomes Analysis"],
            default=["Patient Demographics", "Lab Results"],
            help="Select which types of analysis to perform"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Input")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload Patient IDs",
            type=['csv', 'txt', 'xlsx'],
            help="Upload a file containing patient IDs (one per line or column)"
        )
        
        # Manual input section
        st.subheader("Or Enter Patient IDs Manually")
        patient_ids_input = st.text_area(
            "Patient IDs (one per line or comma-separated)",
            height=100,
            placeholder="Enter patient IDs here...\nExample:\n12345\n67890\n11111"
        )
    
    with col2:
        st.header("📈 Quick Stats")
        st.info("Upload patient IDs to see analysis options")
        
        # Placeholder for stats
        if uploaded_file or patient_ids_input:
            st.success("✅ Data ready for analysis")
        else:
            st.warning("⚠️ No data provided")
    
    # Process patient IDs
    patient_ids = []
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, header=None)
            else:  # txt file
                content = uploaded_file.read().decode('utf-8')
                patient_ids = [line.strip() for line in content.split('\n') if line.strip()]
            
            if 'df' in locals():
                patient_ids = df[0].astype(str).tolist()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
    
    elif patient_ids_input:
        # Parse manual input
        lines = patient_ids_input.strip().split('\n')
        for line in lines:
            if ',' in line:
                # Comma-separated
                ids = [id.strip() for id in line.split(',') if id.strip()]
                patient_ids.extend(ids)
            else:
                # One per line
                if line.strip():
                    patient_ids.append(line.strip())
    
    # Display patient IDs summary
    if patient_ids:
        st.success(f"✅ Found {len(patient_ids)} patient IDs")
        
        # Show sample of patient IDs
        with st.expander("View Patient IDs"):
            st.write("Sample of patient IDs:")
            st.code(patient_ids[:10] if len(patient_ids) > 10 else patient_ids)
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            run_analysis(patient_ids, analysis_type, config)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MSL Data Analysis Dashboard** | "
        "Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

def run_analysis(patient_ids, analysis_types, config):
    """Run the analysis pipeline"""
    
    try:
        # Initialize data fetcher
        with st.spinner("Connecting to database..."):
            data_fetcher = MSLDataFetcher(config)
        
        # Fetch data
        with st.spinner("Fetching patient data..."):
            data = data_fetcher.fetch_patient_data(patient_ids)
        
        if data.empty:
            st.error("No data found for the provided patient IDs")
            return
        
        st.success(f"✅ Retrieved data for {len(data)} patients")
        
        # Initialize plot generator
        plot_generator = PlotGenerator()
        
        # Generate plots based on analysis types
        for analysis_type in analysis_types:
            st.subheader(f"📊 {analysis_type}")
            
            with st.spinner(f"Generating {analysis_type} plots..."):
                if analysis_type == "Patient Demographics":
                    fig = plot_generator.plot_demographics(data)
                    st.pyplot(fig)
                    
                elif analysis_type == "Lab Results":
                    fig = plot_generator.plot_lab_results(data)
                    st.pyplot(fig)
                    
                elif analysis_type == "Treatment History":
                    fig = plot_generator.plot_treatment_history(data)
                    st.pyplot(fig)
                    
                elif analysis_type == "Outcomes Analysis":
                    fig = plot_generator.plot_outcomes(data)
                    st.pyplot(fig)
        
        # Download options
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download data as CSV
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="Download Data (CSV)",
                data=csv_data,
                file_name=f"msl_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download summary report
            summary_report = generate_summary_report(data, patient_ids)
            st.download_button(
                label="Download Summary Report",
                data=summary_report,
                file_name=f"msl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        logger.error(f"Analysis error: {str(e)}")

def generate_summary_report(data, patient_ids):
    """Generate a summary report"""
    report = f"""
MSL Data Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Patient Analysis Summary:
- Total patient IDs provided: {len(patient_ids)}
- Patients with data: {len(data)}
- Data coverage: {len(data)/len(patient_ids)*100:.1f}%

Data Overview:
- Total records: {len(data)}
- Date range: {data.get('date', pd.Series()).min()} to {data.get('date', pd.Series()).max()}
- Columns available: {', '.join(data.columns.tolist())}

Key Statistics:
{data.describe().to_string()}
"""
    return report

if __name__ == "__main__":
    main() 