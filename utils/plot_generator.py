#!/usr/bin/env python3
"""
Plot generator utility for MSL Web Application
Handles creation of various plots and visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PlotGenerator:
    """Plot generator for MSL data visualizations"""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (12, 8), dpi: int = 300):
        """Initialize plot generator with styling"""
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Could not set style {style}: {e}")
            plt.style.use('default')
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = dpi
        
        # Color palette
        self.colors = sns.color_palette("husl", 8)
    
    def plot_demographics(self, data: pd.DataFrame) -> plt.Figure:
        """Generate demographics plots"""
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('Patient Demographics Analysis', fontsize=16, fontweight='bold')
            
            # Extract demographics data
            if 'age' in data.columns:
                # Age distribution
                age_data = data['age'].dropna()
                if not age_data.empty:
                    axes[0, 0].hist(age_data, bins=20, color=self.colors[0], alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('Age Distribution')
                    axes[0, 0].set_xlabel('Age')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].grid(True, alpha=0.3)
            
            if 'gender' in data.columns:
                # Gender distribution
                gender_counts = data['gender'].value_counts()
                if not gender_counts.empty:
                    axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                                  colors=self.colors[1:3])
                    axes[0, 1].set_title('Gender Distribution')
            
            if 'race' in data.columns:
                # Race distribution
                race_counts = data['race'].value_counts().head(10)
                if not race_counts.empty:
                    axes[1, 0].bar(range(len(race_counts)), race_counts.values, color=self.colors[3])
                    axes[1, 0].set_title('Race Distribution (Top 10)')
                    axes[1, 0].set_xlabel('Race')
                    axes[1, 0].set_ylabel('Count')
                    axes[1, 0].set_xticks(range(len(race_counts)))
                    axes[1, 0].set_xticklabels(race_counts.index, rotation=45, ha='right')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # Patient count over time (if date column exists)
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    date_counts = data[date_col].dt.year.value_counts().sort_index()
                    if not date_counts.empty:
                        axes[1, 1].plot(date_counts.index, date_counts.values, marker='o', 
                                       color=self.colors[4], linewidth=2)
                        axes[1, 1].set_title('Patients by Year')
                        axes[1, 1].set_xlabel('Year')
                        axes[1, 1].set_ylabel('Patient Count')
                        axes[1, 1].grid(True, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not plot date data: {e}")
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating demographics plot: {e}")
            return self._create_error_figure("Demographics Plot Error")
    
    def plot_lab_results(self, data: pd.DataFrame) -> plt.Figure:
        """Generate laboratory results plots"""
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('Laboratory Results Analysis', fontsize=16, fontweight='bold')
            
            # Find lab-related columns
            lab_cols = [col for col in data.columns if any(keyword in col.lower() 
                        for keyword in ['test', 'lab', 'value', 'result'])]
            
            if lab_cols:
                # Test value distribution (assuming numeric)
                numeric_cols = data[lab_cols].select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Box plot of test values
                    test_data = data[numeric_cols[:5]]  # Limit to 5 columns
                    test_data_melted = test_data.melt()
                    sns.boxplot(data=test_data_melted, x='variable', y='value', ax=axes[0, 0])
                    axes[0, 0].set_title('Test Values Distribution')
                    axes[0, 0].set_xlabel('Test Type')
                    axes[0, 0].set_ylabel('Value')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Test frequency
                if 'test_name' in data.columns:
                    test_counts = data['test_name'].value_counts().head(10)
                    axes[0, 1].bar(range(len(test_counts)), test_counts.values, color=self.colors[1])
                    axes[0, 1].set_title('Most Common Tests')
                    axes[0, 1].set_xlabel('Test Name')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].set_xticks(range(len(test_counts)))
                    axes[0, 1].set_xticklabels(test_counts.index, rotation=45, ha='right')
                    axes[0, 1].grid(True, alpha=0.3)
            
            # Time series of lab results (if date column exists)
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                try:
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    # Plot trend of one numeric column over time
                    trend_data = data[[date_col, numeric_cols[0]]].dropna()
                    if not trend_data.empty:
                        trend_data = trend_data.sort_values(date_col)
                        axes[1, 0].scatter(trend_data[date_col], trend_data[numeric_cols[0]], 
                                         alpha=0.6, color=self.colors[2])
                        axes[1, 0].set_title(f'{numeric_cols[0]} Over Time')
                        axes[1, 0].set_xlabel('Date')
                        axes[1, 0].set_ylabel(numeric_cols[0])
                        axes[1, 0].tick_params(axis='x', rotation=45)
                        axes[1, 0].grid(True, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not plot time series: {e}")
            
            # Correlation heatmap (if multiple numeric columns)
            if len(numeric_cols) > 1:
                corr_data = data[numeric_cols].corr()
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                           ax=axes[1, 1], cbar_kws={'shrink': 0.8})
                axes[1, 1].set_title('Test Values Correlation')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating lab results plot: {e}")
            return self._create_error_figure("Lab Results Plot Error")
    
    def plot_treatment_history(self, data: pd.DataFrame) -> plt.Figure:
        """Generate treatment history plots"""
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('Treatment History Analysis', fontsize=16, fontweight='bold')
            
            # Treatment type distribution
            treatment_cols = [col for col in data.columns if any(keyword in col.lower() 
                           for keyword in ['treatment', 'medication', 'therapy'])]
            
            if treatment_cols:
                # Treatment type frequency
                if 'treatment_type' in data.columns:
                    treatment_counts = data['treatment_type'].value_counts().head(10)
                    axes[0, 0].bar(range(len(treatment_counts)), treatment_counts.values, color=self.colors[0])
                    axes[0, 0].set_title('Treatment Types')
                    axes[0, 0].set_xlabel('Treatment Type')
                    axes[0, 0].set_ylabel('Count')
                    axes[0, 0].set_xticks(range(len(treatment_counts)))
                    axes[0, 0].set_xticklabels(treatment_counts.index, rotation=45, ha='right')
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Medication frequency
                if 'medication_name' in data.columns:
                    med_counts = data['medication_name'].value_counts().head(10)
                    axes[0, 1].bar(range(len(med_counts)), med_counts.values, color=self.colors[1])
                    axes[0, 1].set_title('Most Common Medications')
                    axes[0, 1].set_xlabel('Medication')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].set_xticks(range(len(med_counts)))
                    axes[0, 1].set_xticklabels(med_counts.index, rotation=45, ha='right')
                    axes[0, 1].grid(True, alpha=0.3)
            
            # Treatment timeline (if date column exists)
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    treatment_timeline = data[date_col].value_counts().sort_index()
                    if not treatment_timeline.empty:
                        axes[1, 0].plot(treatment_timeline.index, treatment_timeline.values, 
                                       marker='o', color=self.colors[2], linewidth=2)
                        axes[1, 0].set_title('Treatment Timeline')
                        axes[1, 0].set_xlabel('Date')
                        axes[1, 0].set_ylabel('Treatment Count')
                        axes[1, 0].tick_params(axis='x', rotation=45)
                        axes[1, 0].grid(True, alpha=0.3)
                except Exception as e:
                    logger.warning(f"Could not plot treatment timeline: {e}")
            
            # Response analysis
            if 'response' in data.columns:
                response_counts = data['response'].value_counts()
                if not response_counts.empty:
                    axes[1, 1].pie(response_counts.values, labels=response_counts.index, 
                                  autopct='%1.1f%%', colors=self.colors[3:6])
                    axes[1, 1].set_title('Treatment Response')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating treatment history plot: {e}")
            return self._create_error_figure("Treatment History Plot Error")
    
    def plot_outcomes(self, data: pd.DataFrame) -> plt.Figure:
        """Generate outcomes analysis plots"""
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            fig.suptitle('Patient Outcomes Analysis', fontsize=16, fontweight='bold')
            
            # Survival analysis (if survival-related columns exist)
            survival_cols = [col for col in data.columns if any(keyword in col.lower() 
                           for keyword in ['survival', 'outcome', 'status', 'mortality'])]
            
            if survival_cols:
                # Outcome distribution
                if 'outcome' in data.columns:
                    outcome_counts = data['outcome'].value_counts()
                    axes[0, 0].pie(outcome_counts.values, labels=outcome_counts.index, 
                                  autopct='%1.1f%%', colors=self.colors[:len(outcome_counts)])
                    axes[0, 0].set_title('Patient Outcomes')
                
                # Status over time
                date_cols = [col for col in data.columns if 'date' in col.lower()]
                if date_cols and 'status' in data.columns:
                    date_col = date_cols[0]
                    try:
                        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                        status_timeline = data.groupby([date_col, 'status']).size().unstack(fill_value=0)
                        if not status_timeline.empty:
                            status_timeline.plot(kind='area', stacked=True, ax=axes[0, 1], 
                                               colors=self.colors[1:4])
                            axes[0, 1].set_title('Status Over Time')
                            axes[0, 1].set_xlabel('Date')
                            axes[0, 1].set_ylabel('Patient Count')
                            axes[0, 1].tick_params(axis='x', rotation=45)
                            axes[0, 1].grid(True, alpha=0.3)
                    except Exception as e:
                        logger.warning(f"Could not plot status timeline: {e}")
            
            # Risk factors analysis (if multiple variables)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Correlation with outcome (if outcome is categorical)
                if 'outcome' in data.columns:
                    try:
                        # Convert outcome to numeric for correlation
                        outcome_numeric = pd.Categorical(data['outcome']).codes
                        correlations = []
                        for col in numeric_cols[:5]:  # Limit to 5 variables
                            corr = np.corrcoef(data[col].dropna(), outcome_numeric[:len(data[col].dropna())])[0, 1]
                            correlations.append((col, corr))
                        
                        # Plot correlations
                        cols, corrs = zip(*correlations)
                        axes[1, 0].bar(range(len(cols)), corrs, color=self.colors[4])
                        axes[1, 0].set_title('Risk Factor Correlations')
                        axes[1, 0].set_xlabel('Variables')
                        axes[1, 0].set_ylabel('Correlation with Outcome')
                        axes[1, 0].set_xticks(range(len(cols)))
                        axes[1, 0].set_xticklabels(cols, rotation=45, ha='right')
                        axes[1, 0].grid(True, alpha=0.3)
                    except Exception as e:
                        logger.warning(f"Could not plot risk factors: {e}")
            
            # Summary statistics
            summary_stats = data.describe()
            if not summary_stats.empty:
                # Create a simple summary plot
                axes[1, 1].text(0.1, 0.9, f"Total Patients: {len(data)}", fontsize=12)
                axes[1, 1].text(0.1, 0.8, f"Data Points: {data.shape[0]}", fontsize=12)
                axes[1, 1].text(0.1, 0.7, f"Variables: {data.shape[1]}", fontsize=12)
                axes[1, 1].text(0.1, 0.6, f"Date Range: {data.get('date', pd.Series()).min()} to {data.get('date', pd.Series()).max()}", fontsize=12)
                axes[1, 1].set_title('Summary Statistics')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating outcomes plot: {e}")
            return self._create_error_figure("Outcomes Plot Error")
    
    def plot_summary_dashboard(self, data: pd.DataFrame) -> plt.Figure:
        """Generate a comprehensive summary dashboard"""
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('MSL Data Summary Dashboard', fontsize=20, fontweight='bold')
            
            # Create a grid of subplots
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Key metrics
            ax1 = fig.add_subplot(gs[0, :2])
            metrics_text = f"""
            Dataset Overview:
            • Total Records: {len(data):,}
            • Unique Patients: {data.get('patient_id', pd.Series()).nunique():,}
            • Variables: {data.shape[1]}
            • Date Range: {data.get('date', pd.Series()).min()} to {data.get('date', pd.Series()).max()}
            • Missing Data: {data.isnull().sum().sum():,} cells
            """
            ax1.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax1.set_title('Key Metrics', fontweight='bold')
            ax1.axis('off')
            
            # Data completeness
            ax2 = fig.add_subplot(gs[0, 2:])
            completeness = (1 - data.isnull().sum() / len(data)) * 100
            completeness = completeness.sort_values(ascending=True)
            ax2.barh(range(len(completeness)), completeness.values, color=self.colors[0])
            ax2.set_yticks(range(len(completeness)))
            ax2.set_yticklabels(completeness.index, fontsize=8)
            ax2.set_xlabel('Completeness (%)')
            ax2.set_title('Data Completeness by Variable', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Variable types
            ax3 = fig.add_subplot(gs[1, :2])
            dtype_counts = data.dtypes.value_counts()
            ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', 
                   colors=self.colors[1:4])
            ax3.set_title('Variable Types', fontweight='bold')
            
            # Sample data preview
            ax4 = fig.add_subplot(gs[1, 2:])
            sample_data = data.head(5).to_string()
            ax4.text(0.05, 0.95, 'Sample Data (First 5 rows):', fontsize=10, fontweight='bold', 
                    transform=ax4.transAxes)
            ax4.text(0.05, 0.85, sample_data, fontsize=6, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace')
            ax4.axis('off')
            
            # Correlation heatmap (if numeric data)
            ax5 = fig.add_subplot(gs[2, :])
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           ax=ax5, cbar_kws={'shrink': 0.8})
                ax5.set_title('Numeric Variables Correlation', fontweight='bold')
            else:
                ax5.text(0.5, 0.5, 'Insufficient numeric data for correlation', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Correlation Analysis', fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {e}")
            return self._create_error_figure("Summary Dashboard Error")
    
    def _create_error_figure(self, error_message: str) -> plt.Figure:
        """Create an error figure when plotting fails"""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.text(0.5, 0.5, f"Error: {error_message}\n\nPlease check your data and try again.", 
                ha='center', va='center', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        ax.set_title('Plot Generation Error', fontweight='bold', color='red')
        ax.axis('off')
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, format: str = 'png') -> str:
        """Save plot to file"""
        try:
            filepath = f"static/plots/{filename}.{format}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Plot saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return "" 