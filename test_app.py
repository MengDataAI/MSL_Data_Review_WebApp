#!/usr/bin/env python3
"""
Test script for MSL data review Web Application
This script tests the basic functionality without requiring a database connection
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def create_sample_data():
    """Create sample data for testing"""
    
    # Sample patient demographics
    patients = pd.DataFrame({
        'patient_id': [f'PAT{i:03d}' for i in range(1, 21)],
        'age': np.random.randint(25, 85, 20),
        'gender': np.random.choice(['Male', 'Female'], 20),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 20),
        'diagnosis_date': [datetime.now() - timedelta(days=np.random.randint(100, 1000)) for _ in range(20)]
    })
    
    # Sample lab results
    lab_results = []
    for patient_id in patients['patient_id']:
        for _ in range(np.random.randint(3, 8)):  # 3-7 lab tests per patient
            lab_results.append({
                'patient_id': patient_id,
                'test_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'test_name': np.random.choice(['Creatinine', 'BUN', 'GFR', 'Albumin', 'Bilirubin']),
                'test_value': np.random.uniform(0.5, 10.0),
                'test_unit': np.random.choice(['mg/dL', 'g/dL', 'mL/min']),
                'abnormal_flag': np.random.choice(['Normal', 'High', 'Low'], p=[0.7, 0.15, 0.15])
            })
    
    lab_df = pd.DataFrame(lab_results)
    
    # Sample treatment history
    treatments = []
    for patient_id in patients['patient_id']:
        for _ in range(np.random.randint(1, 4)):  # 1-3 treatments per patient
            treatments.append({
                'patient_id': patient_id,
                'treatment_date': datetime.now() - timedelta(days=np.random.randint(1, 500)),
                'treatment_type': np.random.choice(['Medication', 'Surgery', 'Therapy']),
                'medication_name': np.random.choice(['Tacrolimus', 'Mycophenolate', 'Prednisone', 'Cyclosporine']),
                'dosage': f"{np.random.randint(1, 10)}mg",
                'response': np.random.choice(['Good', 'Fair', 'Poor'], p=[0.6, 0.3, 0.1])
            })
    
    treatment_df = pd.DataFrame(treatments)
    
    return patients, lab_df, treatment_df

def test_plot_generator():
    """Test the plot generator with sample data"""
    try:
        from plot_generator import PlotGenerator
        
        print("Testing Plot Generator...")
        
        # Create sample data
        patients, lab_df, treatment_df = create_sample_data()
        
        # Initialize plot generator
        plot_gen = PlotGenerator()
        
        # Test demographics plot
        print("  - Testing demographics plot...")
        fig1 = plot_gen.plot_demographics(patients)
        print("    ✓ Demographics plot created successfully")
        
        # Test lab results plot
        print("  - Testing lab results plot...")
        fig2 = plot_gen.plot_lab_results(lab_df)
        print("    ✓ Lab results plot created successfully")
        
        # Test treatment history plot
        print("  - Testing treatment history plot...")
        fig3 = plot_gen.plot_treatment_history(treatment_df)
        print("    ✓ Treatment history plot created successfully")
        
        # Test outcomes plot
        print("  - Testing outcomes plot...")
        fig4 = plot_gen.plot_outcomes(patients)
        print("    ✓ Outcomes plot created successfully")
        
        print("✓ All plot tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Plot generator test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config.settings import load_config
        
        print("Testing Configuration...")
        
        # Test config loading
        config = load_config()
        print(f"  - App name: {config.APP_NAME}")
        print(f"  - App version: {config.APP_VERSION}")
        print(f"  - MSL table: {config.MSL_TABLE}")
        
        print("✓ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_validation():
    """Test data validation functions"""
    try:
        from data_fetcher import MSLDataFetcher
        from config.settings import load_config
        
        print("Testing Data Validation...")
        
        # Create a mock config
        config = load_config()
        
        # Initialize data fetcher
        fetcher = MSLDataFetcher(config)
        
        # Test patient ID validation
        test_ids = ['PAT001', 'PAT002', 'invalid@id', 'PAT003', '', 'PAT004']
        valid_ids = fetcher.validate_patient_ids(test_ids)
        
        print(f"  - Input IDs: {test_ids}")
        print(f"  - Valid IDs: {valid_ids}")
        print(f"  - Filtered out: {len(test_ids) - len(valid_ids)} invalid IDs")
        
        if len(valid_ids) == 4:  # Should filter out invalid@id and empty string
            print("✓ Data validation test passed!")
            return True
        else:
            print("✗ Data validation test failed - unexpected number of valid IDs")
            return False
            
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MSL Web Application - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Data Validation", test_data_validation),
        ("Plot Generator", test_plot_generator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} Test...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application:")
        print("1. Set up your .env file with Databricks credentials")
        print("2. Run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 