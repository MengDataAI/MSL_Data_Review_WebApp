#!/usr/bin/env python3
"""
Data fetcher utility for MSL Web Application
Handles database connections and data retrieval
"""

import pandas as pd
import numpy as np
from databricks import sql
import os
import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

class MSLDataFetcher:
    """Data fetcher for MSL database operations"""
    
    def __init__(self, config):
        """Initialize data fetcher with configuration"""
        self.config = config
        self.connection = None
        
    def connect(self):
        """Establish connection to Databricks"""
        try:
            if not self.config.DATABRICKS_TOKEN:
                raise ValueError("DATABRICKS_TOKEN is required")
            
            self.connection = sql.connect(
                server_hostname=self.config.DATABRICKS_SERVER,
                http_path=self.config.DATABRICKS_HTTP_PATH,
                access_token=self.config.DATABRICKS_TOKEN
            )
            
            logger.info("Successfully connected to Databricks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Databricks: {str(e)}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.connection = None
    
    def validate_patient_ids(self, patient_ids: List[str]) -> List[str]:
        """Validate and sanitize patient IDs"""
        if not patient_ids:
            return []
        
        # Remove duplicates and empty values
        valid_ids = list(set([str(pid).strip() for pid in patient_ids if str(pid).strip()]))
        
        # Basic validation - allow alphanumeric, hyphens, underscores
        pattern = re.compile(r'^[A-Za-z0-9_-]+$')
        filtered_ids = [pid for pid in valid_ids if pattern.match(pid)]
        
        if len(filtered_ids) != len(valid_ids):
            logger.warning(f"Filtered out {len(valid_ids) - len(filtered_ids)} invalid patient IDs")
        
        return filtered_ids
    
    def fetch_patient_data(self, patient_ids: List[str]) -> pd.DataFrame:
        """Fetch patient data from MSL table"""
        try:
            # Validate patient IDs
            valid_ids = self.validate_patient_ids(patient_ids)
            if not valid_ids:
                logger.warning("No valid patient IDs provided")
                return pd.DataFrame()
            
            # Connect to database
            if not self.connection:
                self.connect()
            
            # Build safe query
            placeholders = ', '.join(['%s'] * len(valid_ids))
            query = f"""
                SELECT *
                FROM {self.config.MSL_TABLE}
                WHERE patient_id IN ({placeholders})
                ORDER BY patient_id, date
            """
            
            logger.info(f"Fetching data for {len(valid_ids)} patients")
            
            # Execute query
            with self.connection.cursor() as cursor:
                cursor.execute(query, valid_ids)
                result = cursor.fetchall()
                
                if not result:
                    logger.warning("No data found for the provided patient IDs")
                    return pd.DataFrame()
                
                # Get column names
                columns = [col[0] for col in cursor.description]
                
                # Create DataFrame
                df = pd.DataFrame(result, columns=columns)
                
                logger.info(f"Retrieved {len(df)} records for {len(valid_ids)} patients")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching patient data: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def fetch_patient_demographics(self, patient_ids: List[str]) -> pd.DataFrame:
        """Fetch patient demographics data"""
        try:
            valid_ids = self.validate_patient_ids(patient_ids)
            if not valid_ids:
                return pd.DataFrame()
            
            if not self.connection:
                self.connect()
            
            placeholders = ', '.join(['%s'] * len(valid_ids))
            query = f"""
                SELECT DISTINCT 
                    patient_id,
                    age,
                    gender,
                    race,
                    ethnicity,
                    diagnosis_date,
                    transplant_date
                FROM {self.config.MSL_TABLE}
                WHERE patient_id IN ({placeholders})
                ORDER BY patient_id
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query, valid_ids)
                result = cursor.fetchall()
                
                if not result:
                    return pd.DataFrame()
                
                columns = ['patient_id', 'age', 'gender', 'race', 'ethnicity', 'diagnosis_date', 'transplant_date']
                df = pd.DataFrame(result, columns=columns)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching demographics: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def fetch_lab_results(self, patient_ids: List[str]) -> pd.DataFrame:
        """Fetch laboratory results data"""
        try:
            valid_ids = self.validate_patient_ids(patient_ids)
            if not valid_ids:
                return pd.DataFrame()
            
            if not self.connection:
                self.connect()
            
            placeholders = ', '.join(['%s'] * len(valid_ids))
            query = f"""
                SELECT 
                    patient_id,
                    test_date,
                    test_name,
                    test_value,
                    test_unit,
                    reference_range,
                    abnormal_flag
                FROM {self.config.MSL_TABLE}
                WHERE patient_id IN ({placeholders})
                AND test_name IS NOT NULL
                ORDER BY patient_id, test_date, test_name
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query, valid_ids)
                result = cursor.fetchall()
                
                if not result:
                    return pd.DataFrame()
                
                columns = ['patient_id', 'test_date', 'test_name', 'test_value', 'test_unit', 'reference_range', 'abnormal_flag']
                df = pd.DataFrame(result, columns=columns)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching lab results: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def fetch_treatment_history(self, patient_ids: List[str]) -> pd.DataFrame:
        """Fetch treatment history data"""
        try:
            valid_ids = self.validate_patient_ids(patient_ids)
            if not valid_ids:
                return pd.DataFrame()
            
            if not self.connection:
                self.connect()
            
            placeholders = ', '.join(['%s'] * len(valid_ids))
            query = f"""
                SELECT 
                    patient_id,
                    treatment_date,
                    treatment_type,
                    medication_name,
                    dosage,
                    frequency,
                    duration,
                    response
                FROM {self.config.MSL_TABLE}
                WHERE patient_id IN ({placeholders})
                AND treatment_type IS NOT NULL
                ORDER BY patient_id, treatment_date
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query, valid_ids)
                result = cursor.fetchall()
                
                if not result:
                    return pd.DataFrame()
                
                columns = ['patient_id', 'treatment_date', 'treatment_type', 'medication_name', 'dosage', 'frequency', 'duration', 'response']
                df = pd.DataFrame(result, columns=columns)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching treatment history: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def get_table_schema(self) -> Dict[str, Any]:
        """Get table schema information"""
        try:
            if not self.connection:
                self.connect()
            
            query = f"DESCRIBE {self.config.MSL_TABLE}"
            
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                
                schema = {}
                for row in result:
                    if len(row) >= 2:
                        column_name = row[0]
                        data_type = row[1]
                        schema[column_name] = data_type
                
                return schema
                
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise
        finally:
            self.disconnect()
    
    def get_sample_data(self, limit: int = 5) -> pd.DataFrame:
        """Get sample data for testing"""
        try:
            if not self.connection:
                self.connect()
            
            query = f"""
                SELECT *
                FROM {self.config.MSL_TABLE}
                LIMIT {limit}
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                
                if not result:
                    return pd.DataFrame()
                
                columns = [col[0] for col in cursor.description]
                df = pd.DataFrame(result, columns=columns)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting sample data: {str(e)}")
            raise
        finally:
            self.disconnect() 