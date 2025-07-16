#!/usr/bin/env python3
"""
Configuration settings for MSL data review Web Application
"""

import os
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for MSL Web Application"""
    
    def __init__(self):
        # Databricks configuration
        self.DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER", "adb-5253398580030522.2.azuredatabricks.net")
        self.DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/10532adbe7a19d7d")
        self.DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
        
        # Database configuration
        self.DATABASE_CATALOG = os.getenv("DATABASE_CATALOG", "production")
        self.DATABASE_SCHEMA = os.getenv("DATABASE_SCHEMA", "default")
        self.MSL_TABLE = os.getenv("MSL_TABLE", "nglims.v_msl")
        
        # Application configuration
        self.APP_NAME = "MSL Data Review"
        self.APP_VERSION = "1.0.0"
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        
        # File upload configuration
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        self.ALLOWED_EXTENSIONS = {'.csv', '.txt', '.xlsx', '.xls'}
        
        # Plot configuration
        self.PLOT_STYLE = "seaborn-v0_8"
        self.PLOT_FIGSIZE = (12, 8)
        self.PLOT_DPI = 300
        
        # Cache configuration
        self.CACHE_TTL = 3600  # 1 hour
        
        # Security configuration
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        
    def validate(self) -> bool:
        """Validate configuration"""
        required_vars = [
            "DATABRICKS_TOKEN"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(self, var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "databricks_server": self.DATABRICKS_SERVER,
            "databricks_http_path": self.DATABRICKS_HTTP_PATH,
            "database_catalog": self.DATABASE_CATALOG,
            "database_schema": self.DATABASE_SCHEMA,
            "msl_table": self.MSL_TABLE,
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "debug": self.DEBUG,
            "max_file_size": self.MAX_FILE_SIZE,
            "allowed_extensions": list(self.ALLOWED_EXTENSIONS),
            "plot_style": self.PLOT_STYLE,
            "plot_figsize": self.PLOT_FIGSIZE,
            "plot_dpi": self.PLOT_DPI,
            "cache_ttl": self.CACHE_TTL
        }

def load_config() -> Config:
    """Load configuration from environment and config files"""
    config = Config()
    
    # Load from config file if exists
    config_file = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Update config with file values
            for key, value in file_config.items():
                if hasattr(config, key.upper()):
                    setattr(config, key.upper(), value)
                    
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        raise ValueError("Invalid configuration")
    
    return config

def save_config(config: Config, filepath: str = None) -> bool:
    """Save configuration to file"""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "config.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Could not save config file: {e}")
        return False

# Default configuration instance
default_config = load_config() 