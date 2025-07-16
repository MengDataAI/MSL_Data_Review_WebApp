#!/usr/bin/env python3
"""
Data fetcher utility for MSL Data Review Web Application
Handles database connections, SQL generation, and data retrieval
"""

import pandas as pd
from databricks import sql
import logging
import re

logger = logging.getLogger(__name__)

class MSLDataFetcher:
    """Data fetcher for MSL database operations"""

    def __init__(self, config):
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

    def parse_patient_ids(self, patient_ids_raw):
        """Parse and sanitize patient IDs from input string"""
        # Split by comma, newline, or whitespace, remove empty, strip spaces
        ids = re.split(r'[\s,]+', patient_ids_raw)
        ids = [pid.strip() for pid in ids if pid.strip()]
        # Remove duplicates and basic validation (alphanumeric, hyphen, underscore)
        pattern = re.compile(r'^[A-Za-z0-9_-]+$')
        ids = list({pid for pid in ids if pattern.match(pid)})
        return ids

    def fetch_custom_patient_data(self, patient_ids, start_date, end_date):
        """Generate and run the custom SQL query"""
        if not patient_ids:
            logger.warning("No valid patient IDs provided")
            return pd.DataFrame()

        # Format patient IDs for SQL array
        patient_id_list = ',\n            '.join([f"'{pid}'" for pid in patient_ids])

        # Build SQL query
        sql_query = f"""
WITH cte AS (
  SELECT
    DISTINCT LEFT(sf_center_account_code, 4) AS ACCOUNT_CODE,
    transplant_center AS TRANSPLANT_CENTER,
    accessionid AS ACCESSIONID,
    patientid AS PID,
    SEX,
    tx_date_updated AS TRANSPLANT_DATE,
    BIRTHDATE,
    DATEDIFF(current_date(), tx_date_updated) AS TIME_FROM_TRANSPLANT,
    DATEDIFF(tx_date_updated, BIRTHDATE) AS TRANSPLANT_AGE_IN_DAYS,
    ROUND(DATEDIFF(tx_date_updated, BIRTHDATE)/365.25) AS TRANSPLANT_AGE,
    DATEDIFF(COALESCE(draw_date, res_last_release_date), tx_date_updated) AS RESULT_POST_TRANSPLANT,
    prescriber_name AS PRESCRIBER_NAME,
    draw_date AS DRAW_DATE,
    trf_study as TRF_STUDY_TYPE,
    trf_received_date AS DATE_RECEIVED,
    transplanted_organ AS ORGAN,
    donortype AS DONOR_TYPE,
    donor_relation AS DONOR_RELATION,
    product_type AS PRODUCT_TYPE,
    res_init_release_date AS RES_INIT_RELEASE_DATE,
    res_last_release_date AS RES_FINAL_RELEASE_DATE,
    res_result AS RES_RESULT_STRING,
    COALESCE(draw_date, res_last_release_date) AS DATE_FILTER,
    (
      CASE
        WHEN res_result LIKE '<%' THEN CAST(regexp_replace(res_result, '[<%]', '') AS DOUBLE) - 0.01 
        WHEN res_result LIKE '>%' THEN CAST(regexp_replace(res_result, '[>%]', '') AS DOUBLE) + 0.01 
        ELSE CAST(regexp_replace(res_result, '%', '') AS DOUBLE)
      END
    ) AS RES_RESULT_NUMERIC,
    CASE
      WHEN array_contains(
        array(
            {patient_id_list}
        ),
        patientid
      ) THEN 'Y'
      ELSE 'N'
    END AS ON_PROTOCOL
  FROM
    production.data_engineering.nglims_all_tests_updated
  WHERE
    res_result NOT IN ('Unacceptable Sample', 'No Result', 'Not Detected')
    AND res_result IS NOT NULL
  ORDER BY
    PID,
    RESULT_POST_TRANSPLANT
)
SELECT
  ACCOUNT_CODE,
  TRANSPLANT_CENTER,
  ACCESSIONID,
  PID,
  SEX,
  TRANSPLANT_DATE,
  BIRTHDATE,
  TIME_FROM_TRANSPLANT,
  TRANSPLANT_AGE_IN_DAYS,
  TRANSPLANT_AGE,
  RESULT_POST_TRANSPLANT,
  PRESCRIBER_NAME,
  DRAW_DATE,
  TRF_STUDY_TYPE,
  DATE_RECEIVED,
  ORGAN,
  DONOR_TYPE,
  DONOR_RELATION,
  PRODUCT_TYPE,
  RES_INIT_RELEASE_DATE,
  RES_FINAL_RELEASE_DATE,
  DATE_FILTER,
  RES_RESULT_STRING,
  RES_RESULT_NUMERIC,
  ON_PROTOCOL,
  (
    CASE
      WHEN RES_RESULT_NUMERIC >= 1.0 THEN 'HR'
      WHEN (RES_RESULT_NUMERIC < 1.0 AND RES_RESULT_NUMERIC >= 0.5) THEN 'MR'
      WHEN RES_RESULT_NUMERIC < 0.5 THEN 'LR'
      ELSE NULL
    END
  ) AS RISK_LEVEL,
  (
    CASE
      WHEN RESULT_POST_TRANSPLANT < 46 AND RESULT_POST_TRANSPLANT > 0 THEN 'M1'
      WHEN RESULT_POST_TRANSPLANT >= 46 AND RESULT_POST_TRANSPLANT < 76 THEN 'M2'
      WHEN RESULT_POST_TRANSPLANT >= 76 AND RESULT_POST_TRANSPLANT < 107 THEN 'M3'
      WHEN RESULT_POST_TRANSPLANT >= 107 AND RESULT_POST_TRANSPLANT < 153 THEN 'M4'
      WHEN RESULT_POST_TRANSPLANT >= 153 AND RESULT_POST_TRANSPLANT < 230 THEN 'M6'
      WHEN RESULT_POST_TRANSPLANT >= 230 AND RESULT_POST_TRANSPLANT < 320 THEN 'M9'
      WHEN RESULT_POST_TRANSPLANT >= 320 AND RESULT_POST_TRANSPLANT < 395 THEN 'M12'
      ELSE NULL
    END
  ) AS PROTOCOL_TESTING_MONTH
FROM
  cte
WHERE ON_PROTOCOL = 'Y'
AND DATE_FILTER BETWEEN '{start_date}' AND '{end_date}';
"""
        # Connect and run
        try:
            if not self.connection:
                self.connect()
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query)
                result = cursor.fetchall()
                if not result:
                    logger.warning("No data found for the provided query")
                    return pd.DataFrame()
                columns = [col[0] for col in cursor.description]
                df = pd.DataFrame(result, columns=columns)
                return df
        except Exception as e:
            logger.error(f"Error running custom patient data query: {str(e)}")
            raise
        finally:
            self.disconnect() 