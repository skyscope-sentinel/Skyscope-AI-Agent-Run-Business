#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Sentinel Intelligence AI Platform - Deployment Validation System

This module provides comprehensive deployment validation for the Skyscope platform,
ensuring that all components are properly configured and ready for production use.
It performs system checks, configuration validation, connection testing, security
verification, and generates detailed deployment reports.

Created on: July 16, 2025
Author: Skyscope Sentinel Intelligence
"""

import os
import sys
import json
import time
import uuid
import socket
import logging
import platform
import argparse
import datetime
import subprocess
import threading
import ssl
import re
import hashlib
import tempfile
import shutil
import psutil
import requests
import urllib.request
import importlib
import configparser
import sqlite3
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment_validation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DeploymentValidator')

# Constants
VALIDATOR_VERSION = "1.0.0"
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_CONFIG_PATH = "config/deployment_validation.json"
DEFAULT_REPORT_DIR = "deployment_reports"

# Validation status
class ValidationStatus(Enum):
    """Enumeration of validation statuses"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class ValidationResult:
    """Data class for storing validation results"""
    name: str
    category: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    duration: float = 0.0

@dataclass
class CategoryResult:
    """Data class for storing category validation results"""
    name: str
    results: List[ValidationResult] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    end_time: Optional[str] = None
    duration: float = 0.0
    
    @property
    def status(self) -> ValidationStatus:
        """Overall status for the category"""
        if not self.results:
            return ValidationStatus.SKIPPED
        
        if any(r.status == ValidationStatus.FAILED for r in self.results):
            return ValidationStatus.FAILED
        
        if all(r.status == ValidationStatus.PASSED for r in self.results):
            return ValidationStatus.PASSED
        
        if any(r.status == ValidationStatus.WARNING for r in self.results):
            return ValidationStatus.WARNING
        
        return ValidationStatus.PASSED

@dataclass
class ValidationConfig:
    """Configuration for deployment validation"""
    check_system: bool = True
    check_config: bool = True
    check_database: bool = True
    check_api: bool = True
    check_security: bool = True
    check_wallets: bool = True
    check_agents: bool = True
    check_income: bool = True
    check_monitoring: bool = True
    check_health: bool = True
    check_backup: bool = True
    check_recovery: bool = True
    check_compliance: bool = True
    
    report_dir: str = DEFAULT_REPORT_DIR
    timeout: int = DEFAULT_TIMEOUT
    verbose: bool = False
    fix_issues: bool = False
    
    install_dir: Optional[str] = None
    config_dir: Optional[str] = None
    data_dir: Optional[str] = None
    
    database_type: str = "sqlite"  # sqlite, mysql, postgresql
    database_connection: Dict[str, str] = field(default_factory=lambda: {
        "host": "localhost",
        "port": "3306",
        "user": "skyscope",
        "password": "",
        "database": "skyscope"
    })
    
    api_endpoints: List[str] = field(default_factory=lambda: [
        "http://localhost:8000/api/status",
        "http://localhost:8000/api/agents",
        "http://localhost:8000/api/wallets"
    ])
    
    min_requirements: Dict[str, Any] = field(default_factory=lambda: {
        "cpu_cores": 4,
        "ram_gb": 8,
        "disk_space_gb": 20,
        "python_version": "3.8.0"
    })
    
    backup_locations: List[str] = field(default_factory=list)
    compliance_rules: List[str] = field(default_factory=list)

class DeploymentValidator:
    """
    Main deployment validation class for Skyscope Sentinel Intelligence AI Platform.
    
    This class orchestrates the validation process across all system components
    and generates comprehensive reports on deployment readiness.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the deployment validator.
        
        Args:
            config: Validation configuration (optional)
        """
        self.config = config or ValidationConfig()
        self.results: List[CategoryResult] = []
        self.start_time = None
        self.end_time = None
        self.total_duration = 0.0
        
        # Create report directory if it doesn't exist
        os.makedirs(self.config.report_dir, exist_ok=True)
        
        # Initialize validators
        self.system_validator = SystemValidator(self.config)
        self.config_validator = ConfigValidator(self.config)
        self.database_validator = DatabaseValidator(self.config)
        self.api_validator = ApiValidator(self.config)
        self.security_validator = SecurityValidator(self.config)
        self.wallet_validator = WalletValidator(self.config)
        self.agent_validator = AgentValidator(self.config)
        self.income_validator = IncomeValidator(self.config)
        self.monitoring_validator = MonitoringValidator(self.config)
        self.health_validator = HealthValidator(self.config)
        self.backup_validator = BackupValidator(self.config)
        self.recovery_validator = RecoveryValidator(self.config)
        self.compliance_validator = ComplianceValidator(self.config)
        
        logger.info(f"Deployment validator initialized with configuration")
    
    @classmethod
    def from_config_file(cls, config_path: str = DEFAULT_CONFIG_PATH) -> 'DeploymentValidator':
        """
        Create a deployment validator instance from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            DeploymentValidator instance
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            config = ValidationConfig(**config_data)
            return cls(config)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def validate_deployment(self) -> bool:
        """
        Run all deployment validations based on configuration.
        
        Returns:
            True if all validations passed, False otherwise
        """
        self.start_time = time.time()
        logger.info("Starting deployment validation")
        
        # Run validations
        self._run_validations()
        
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        # Generate report
        self._generate_report()
        
        # Determine overall success
        all_passed = all(
            result.status != ValidationStatus.FAILED for result in self.results
        )
        
        status = "PASSED" if all_passed else "FAILED"
        logger.info(f"Deployment validation completed. Status: {status}")
        logger.info(f"Total duration: {self.total_duration:.2f} seconds")
        
        return all_passed
    
    def _run_validations(self) -> None:
        """Run all validations based on configuration."""
        # System prerequisites
        if self.config.check_system:
            self.results.append(self.system_validator.validate())
        
        # Configuration validation
        if self.config.check_config:
            self.results.append(self.config_validator.validate())
        
        # Database connections
        if self.config.check_database:
            self.results.append(self.database_validator.validate())
        
        # API endpoints
        if self.config.check_api:
            self.results.append(self.api_validator.validate())
        
        # Security settings
        if self.config.check_security:
            self.results.append(self.security_validator.validate())
        
        # Wallet encryption
        if self.config.check_wallets:
            self.results.append(self.wallet_validator.validate())
        
        # Agent deployment
        if self.config.check_agents:
            self.results.append(self.agent_validator.validate())
        
        # Income modules
        if self.config.check_income:
            self.results.append(self.income_validator.validate())
        
        # Monitoring systems
        if self.config.check_monitoring:
            self.results.append(self.monitoring_validator.validate())
        
        # Health checks
        if self.config.check_health:
            self.results.append(self.health_validator.validate())
        
        # Backup systems
        if self.config.check_backup:
            self.results.append(self.backup_validator.validate())
        
        # Disaster recovery
        if self.config.check_recovery:
            self.results.append(self.recovery_validator.validate())
        
        # Compliance requirements
        if self.config.check_compliance:
            self.results.append(self.compliance_validator.validate())
    
    def _generate_report(self) -> None:
        """Generate deployment validation reports in various formats."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        json_report_path = os.path.join(self.config.report_dir, f"deployment_report_{timestamp}.json")
        self._generate_json_report(json_report_path)
        
        # Generate HTML report
        html_report_path = os.path.join(self.config.report_dir, f"deployment_report_{timestamp}.html")
        self._generate_html_report(html_report_path)
        
        # Generate summary text report
        summary_report_path = os.path.join(self.config.report_dir, f"deployment_summary_{timestamp}.txt")
        self._generate_summary_report(summary_report_path)
        
        logger.info(f"Reports generated in {self.config.report_dir}")
    
    def _generate_json_report(self, path: str) -> None:
        """
        Generate a JSON deployment report.
        
        Args:
            path: Path to save the report
        """
        # Convert results to serializable format
        serializable_results = []
        for category_result in self.results:
            serializable_category = {
                "name": category_result.name,
                "start_time": category_result.start_time,
                "end_time": category_result.end_time,
                "duration": category_result.duration,
                "status": category_result.status.value,
                "results": []
            }
            
            for validation_result in category_result.results:
                serializable_validation = {
                    "name": validation_result.name,
                    "category": validation_result.category,
                    "status": validation_result.status.value,
                    "message": validation_result.message,
                    "details": validation_result.details,
                    "timestamp": validation_result.timestamp,
                    "duration": validation_result.duration
                }
                serializable_category["results"].append(serializable_validation)
            
            serializable_results.append(serializable_category)
        
        # Count results by status
        status_counts = {
            "passed": sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.PASSED),
            "failed": sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.FAILED),
            "warning": sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.WARNING),
            "skipped": sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.SKIPPED),
            "not_applicable": sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.NOT_APPLICABLE)
        }
        
        # Create the report
        report = {
            "validator_version": VALIDATOR_VERSION,
            "timestamp": datetime.datetime.now().isoformat(),
            "system_info": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(logical=False),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            },
            "config": {
                "check_system": self.config.check_system,
                "check_config": self.config.check_config,
                "check_database": self.config.check_database,
                "check_api": self.config.check_api,
                "check_security": self.config.check_security,
                "check_wallets": self.config.check_wallets,
                "check_agents": self.config.check_agents,
                "check_income": self.config.check_income,
                "check_monitoring": self.config.check_monitoring,
                "check_health": self.config.check_health,
                "check_backup": self.config.check_backup,
                "check_recovery": self.config.check_recovery,
                "check_compliance": self.config.check_compliance,
                "timeout": self.config.timeout,
                "fix_issues": self.config.fix_issues
            },
            "summary": {
                "start_time": datetime.datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "total_duration": self.total_duration,
                "total_checks": sum(len(cat.results) for cat in self.results),
                "status_counts": status_counts,
                "overall_status": "PASSED" if status_counts["failed"] == 0 else "FAILED"
            },
            "results": serializable_results
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_html_report(self, path: str) -> None:
        """
        Generate an HTML deployment report.
        
        Args:
            path: Path to save the report
        """
        # Simple HTML template for the report
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skyscope Deployment Validation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            background-color: #34495e;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .category {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }
        .category-header {
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
        }
        .category-header.passed {
            background-color: #d4edda;
        }
        .category-header.failed {
            background-color: #f8d7da;
        }
        .category-header.warning {
            background-color: #fff3cd;
        }
        .category-body {
            padding: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .passed {
            color: #28a745;
        }
        .failed {
            color: #dc3545;
        }
        .warning {
            color: #ffc107;
        }
        .skipped {
            color: #6c757d;
        }
        .not_applicable {
            color: #6c757d;
        }
        .details-toggle {
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }
        .details {
            display: none;
            background-color: #f8f9fa;
            padding: 10px;
            margin-top: 5px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            color: white;
        }
        .status-badge.passed {
            background-color: #28a745;
        }
        .status-badge.failed {
            background-color: #dc3545;
        }
        .status-badge.warning {
            background-color: #ffc107;
            color: #212529;
        }
        .status-badge.skipped {
            background-color: #6c757d;
        }
        .status-badge.not_applicable {
            background-color: #6c757d;
        }
        .system-info {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .system-info-item {
            background-color: #e9ecef;
            padding: 5px 10px;
            border-radius: 3px;
        }
    </style>
    <script>
        function toggleDetails(id) {
            var details = document.getElementById(id);
            if (details.style.display === "block") {
                details.style.display = "none";
            } else {
                details.style.display = "block";
            }
        }
    </script>
</head>
<body>
    <div class="header">
        <h1>Skyscope Sentinel Intelligence AI Platform</h1>
        <h2>Deployment Validation Report</h2>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>Deployment Summary</h2>
        <p>Validator Version: {validator_version}</p>
        <p>Total Duration: {total_duration:.2f} seconds</p>
        <p>Total Checks: {total_checks}</p>
        <p>Overall Status: <span class="status-badge {overall_status_lower}">{overall_status}</span></p>
        <p>Results: 
            <span class="passed">{passed_count} Passed</span> | 
            <span class="failed">{failed_count} Failed</span> | 
            <span class="warning">{warning_count} Warnings</span> | 
            <span class="skipped">{skipped_count} Skipped</span>
        </p>
        
        <h3>System Information</h3>
        <div class="system-info">
            <div class="system-info-item">Hostname: {hostname}</div>
            <div class="system-info-item">Platform: {platform}</div>
            <div class="system-info-item">Python: {python_version}</div>
            <div class="system-info-item">CPU Cores: {cpu_count}</div>
            <div class="system-info-item">Memory: {memory_gb} GB</div>
        </div>
    </div>
    
    <h2>Validation Categories</h2>
    
    {category_results}
    
    <h2>Configuration</h2>
    <table>
        <tr><th>Setting</th><th>Value</th></tr>
        {config_rows}
    </table>
    
    <div style="margin-top: 30px; text-align: center; color: #777;">
        <p>Skyscope Sentinel Intelligence AI Platform - Deployment Validator v{validator_version}</p>
    </div>
</body>
</html>
"""
        
        # Generate category results HTML
        category_results_html = ""
        for category_result in self.results:
            category_status_class = category_result.status.value
            
            category_html = f"""
    <div class="category">
        <div class="category-header {category_status_class}">
            <h3>{category_result.name} <span class="status-badge {category_status_class}">{category_result.status.value.upper()}</span></h3>
            <p>Duration: {category_result.duration:.2f} seconds</p>
        </div>
        <div class="category-body">
            <table>
                <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Message</th>
                    <th>Details</th>
                </tr>
"""
            
            for i, validation_result in enumerate(category_result.results):
                status_class = validation_result.status.value
                status_text = validation_result.status.value.upper()
                details_id = f"details_{category_result.name.lower().replace(' ', '_')}_{i}"
                
                details_content = "<pre>" + json.dumps(validation_result.details, indent=2) + "</pre>" if validation_result.details else ""
                
                validation_html = f"""
                <tr>
                    <td>{validation_result.name}</td>
                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                    <td>{validation_result.message}</td>
                    <td>
                        <span class="details-toggle" onclick="toggleDetails('{details_id}')">Show Details</span>
                        <div id="{details_id}" class="details">
                            {details_content}
                        </div>
                    </td>
                </tr>
"""
                category_html += validation_html
            
            category_html += """
            </table>
        </div>
    </div>
"""
            category_results_html += category_html
        
        # Generate config rows HTML
        config_rows_html = ""
        for key, value in vars(self.config).items():
            if key == "database_connection" and isinstance(value, dict):
                # Mask password
                if "password" in value:
                    masked_value = value.copy()
                    if masked_value["password"]:
                        masked_value["password"] = "********"
                    value_str = str(masked_value)
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
            
            config_rows_html += f"<tr><td>{key}</td><td>{value_str}</td></tr>\n"
        
        # Count results by status
        passed_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.PASSED)
        failed_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.FAILED)
        warning_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.WARNING)
        skipped_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.SKIPPED)
        
        # Determine overall status
        overall_status = "PASSED" if failed_count == 0 else "FAILED"
        overall_status_lower = overall_status.lower()
        
        # System info
        hostname = socket.gethostname()
        platform_str = platform.platform()
        python_version = platform.python_version()
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        # Fill in the template
        html_content = html_template.format(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            validator_version=VALIDATOR_VERSION,
            total_duration=self.total_duration,
            total_checks=sum(len(cat.results) for cat in self.results),
            overall_status=overall_status,
            overall_status_lower=overall_status_lower,
            passed_count=passed_count,
            failed_count=failed_count,
            warning_count=warning_count,
            skipped_count=skipped_count,
            hostname=hostname,
            platform=platform_str,
            python_version=python_version,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            category_results=category_results_html,
            config_rows=config_rows_html
        )
        
        with open(path, 'w') as f:
            f.write(html_content)
    
    def _generate_summary_report(self, path: str) -> None:
        """
        Generate a plain text summary report.
        
        Args:
            path: Path to save the report
        """
        with open(path, 'w') as f:
            f.write("===============================================\n")
            f.write("SKYSCOPE SENTINEL INTELLIGENCE AI PLATFORM\n")
            f.write("DEPLOYMENT VALIDATION SUMMARY\n")
            f.write("===============================================\n\n")
            
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Validator Version: {VALIDATOR_VERSION}\n\n")
            
            f.write("OVERALL SUMMARY\n")
            f.write("---------------\n")
            f.write(f"Total Duration: {self.total_duration:.2f} seconds\n")
            total_checks = sum(len(cat.results) for cat in self.results)
            f.write(f"Total Checks: {total_checks}\n")
            
            # Count results by status
            passed_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.PASSED)
            failed_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.FAILED)
            warning_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.WARNING)
            skipped_count = sum(1 for cat in self.results for res in cat.results if res.status == ValidationStatus.SKIPPED)
            
            f.write(f"Passed: {passed_count}\n")
            f.write(f"Failed: {failed_count}\n")
            f.write(f"Warnings: {warning_count}\n")
            f.write(f"Skipped: {skipped_count}\n\n")
            
            overall_status = "PASSED" if failed_count == 0 else "FAILED"
            f.write(f"Overall Status: {overall_status}\n\n")
            
            f.write("SYSTEM INFORMATION\n")
            f.write("-----------------\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"Python Version: {platform.python_version()}\n")
            f.write(f"CPU Cores: {psutil.cpu_count(logical=False)}\n")
            f.write(f"Memory: {round(psutil.virtual_memory().total / (1024**3), 2)} GB\n\n")
            
            f.write("CATEGORY RESULTS\n")
            f.write("----------------\n")
            for category_result in self.results:
                f.write(f"{category_result.name}: {category_result.status.value.upper()}\n")
                f.write(f"  Duration: {category_result.duration:.2f} seconds\n")
                f.write(f"  Checks: {len(category_result.results)}\n")
                
                category_passed = sum(1 for res in category_result.results if res.status == ValidationStatus.PASSED)
                category_failed = sum(1 for res in category_result.results if res.status == ValidationStatus.FAILED)
                category_warning = sum(1 for res in category_result.results if res.status == ValidationStatus.WARNING)
                
                f.write(f"  Passed: {category_passed}, Failed: {category_failed}, Warnings: {category_warning}\n")
                
                if category_failed > 0:
                    f.write("  Failed Checks:\n")
                    for result in category_result.results:
                        if result.status == ValidationStatus.FAILED:
                            f.write(f"    - {result.name}: {result.message}\n")
                
                if category_warning > 0:
                    f.write("  Warnings:\n")
                    for result in category_result.results:
                        if result.status == ValidationStatus.WARNING:
                            f.write(f"    - {result.name}: {result.message}\n")
                
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("---------------\n")
            if failed_count > 0:
                f.write("The following issues must be resolved before deployment:\n")
                for category_result in self.results:
                    for result in category_result.results:
                        if result.status == ValidationStatus.FAILED:
                            f.write(f"  - [{category_result.name}] {result.name}: {result.message}\n")
            else:
                f.write("No critical issues found. System is ready for deployment.\n")
            
            if warning_count > 0:
                f.write("\nThe following warnings should be addressed:\n")
                for category_result in self.results:
                    for result in category_result.results:
                        if result.status == ValidationStatus.WARNING:
                            f.write(f"  - [{category_result.name}] {result.name}: {result.message}\n")
            
            f.write("\n===============================================\n")

class BaseValidator:
    """Base class for all validators"""
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize the validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self) -> CategoryResult:
        """
        Run the validation.
        
        Returns:
            CategoryResult with validation results
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def _create_validation_result(self, name: str, category: str, status: ValidationStatus, 
                                 message: str, details: Optional[Dict[str, Any]] = None,
                                 duration: float = 0.0) -> ValidationResult:
        """
        Create a validation result.
        
        Args:
            name: Validation name
            category: Validation category
            status: Validation status
            message: Validation message
            details: Additional validation details
            duration: Validation duration in seconds
            
        Returns:
            ValidationResult object
        """
        return ValidationResult(
            name=name,
            category=category,
            status=status,
            message=message,
            details=details or {},
            duration=duration
        )

class SystemValidator(BaseValidator):
    """Validator for system prerequisites"""
    
    def validate(self) -> CategoryResult:
        """
        Validate system prerequisites.
        
        Returns:
            CategoryResult with validation results
        """
        start_time = time.time()
        
        category_result = CategoryResult(name="System Prerequisites")
        
        self.logger.info("Validating system prerequisites")
        
        # Define validation checks
        validations = [
            self._check_cpu,
            self._check_memory,
            self._check_disk_space,
            self._check_python_version,
            self._check_required_packages,
            self._check_network_connectivity,
            self._check_file_permissions,
            self._check_os_compatibility,
            self._check_gpu_availability,
            self._check_system_time
        ]
        
        # Run each validation check
        for validation_func in validations:
            try:
                validation_name = validation_func.__name__.replace("_check_", "")
                self.logger.info(f"Checking {validation_name}")
                
                start = time.time()
                result = validation_func()
                duration = time.time() - start
                
                result.duration = duration
                category_result.results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error during {validation_func.__name__}: {e}")
                
                # Create error result
                error_result = self._create_validation_result(
                    name=validation_func.__name__.replace("_check_", ""),
                    category="System",
                    status=ValidationStatus.FAILED,
                    message=f"Error during validation: {str(e)}",
                    duration=0.0
                )
                
                category_result.results.append(error_result)
        
        # Calculate total duration
        end_time = time.time()
        category_result.duration = end_time - start_time
        category_result.end_time = datetime.datetime.now().isoformat()
        
        self.logger.info(f"System prerequisites validation completed")
        
        return category_result
    
    def _check_cpu(self) -> ValidationResult:
        """Check CPU requirements"""
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        
        min_cores = self.config.min_requirements.get("cpu_cores", 4)
        
        if cpu_count >= min_cores:
            status = ValidationStatus.PASSED
            message = f"System has {cpu_count} physical cores ({cpu_logical} logical), meeting the minimum requirement of {min_cores}"
        else:
            status = ValidationStatus.WARNING
            message = f"System has only {cpu_count} physical cores ({cpu_logical} logical), below the recommended {min_cores}"
        
        # Get CPU details
        cpu_info = {}
        try:
            if platform.system() == "Windows":
                cpu_info["name"] = platform.processor()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info["name"] = line.split(":")[1].strip()
                            break
            elif platform.system() == "Darwin":  # macOS
                cpu_info["name"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            cpu_info["name"] = "Unknown"
        
        # Get CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["current_freq_mhz"] = cpu_freq.current
                cpu_info["min_freq_mhz"] = cpu_freq.min
                cpu_info["max_freq_mhz"] = cpu_freq.max
        except Exception:
            pass
        
        return self._create_validation_result(
            name="cpu",
            category="System",
            status=status,
            message=message,
            details={
                "physical_cores": cpu_count,
                "logical_cores": cpu_logical,
                "min_required": min_cores,
                "cpu_info": cpu_info
            }
        )
    
    def _check_memory(self) -> ValidationResult:
        """Check memory requirements"""
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_percent = mem.percent
        
        min_ram_gb = self.config.min_requirements.get("ram_gb", 8)
        
        if total_gb >= min_ram_gb:
            if available_gb >= min_ram_gb / 2:
                status = ValidationStatus.PASSED
                message = f"System has {total_gb:.2f} GB total RAM with {available_gb:.2f} GB available"
            else:
                status = ValidationStatus.WARNING
                message = f"System has {total_gb:.2f} GB total RAM but only {available_gb:.2f} GB available"
        else:
            status = ValidationStatus.FAILED
            message = f"System has only {total_gb:.2f} GB RAM, below the required {min_ram_gb} GB"
        
        return self._create_validation_result(
            name="memory",
            category="System",
            status=status,
            message=message,
            details={
                "total_gb": round(total_gb, 2),
                "available_gb": round(available_gb, 2),
                "used_percent": used_percent,
                "min_required_gb": min_ram_gb
            }
        )
    
    def _check_disk_space(self) -> ValidationResult:
        """Check disk space requirements"""
        install_dir = self.config.install_dir or os.getcwd()
        
        # Get disk usage for the installation directory
        disk = psutil.disk_usage(install_dir)
        total_gb = disk.total / (1024**3)
        free_gb = disk.free / (1024**3)
        used_percent = disk.percent
        
        min_disk_gb = self.config.min_requirements.get("disk_space_gb", 20)
        
        if free_gb >= min_disk_gb:
            status = ValidationStatus.PASSED
            message = f"System has {free_gb:.2f} GB free disk space, meeting the minimum requirement of {min_disk_gb} GB"
        elif free_gb >= min_disk_gb / 2:
            status = ValidationStatus.WARNING
            message = f"System has only {free_gb:.2f} GB free disk space, approaching the minimum requirement of {min_disk_gb} GB"
        else:
            status = ValidationStatus.FAILED
            message = f"System has only {free_gb:.2f} GB free disk space, below the required {min_disk_gb} GB"
        
        return self._create_validation_result(
            name="disk_space",
            category="System",
            status=status,
            message=message,
            details={
                "install_dir": install_dir,
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "used_percent": used_percent,
                "min_required_gb": min_disk_gb
            }
        )
    
    def _check_python_version(self) -> ValidationResult:
        """Check Python version"""
        current_version = platform.python_version()
        min_version = self.config.min_requirements.get("python_version", "3.8.0")
        
        # Compare versions
        current_parts = [int(x) for x in current_version.split(".")]
        min_parts = [int(x) for x in min_version.split(".")]
        
        # Pad with zeros if needed
        while len(current_parts) < 3:
            current_parts.append(0)
        while len(min_parts) < 3:
            min_parts.append(0)
        
        # Compare version parts
        meets_requirement = False
        for i in range(3):
            if current_parts[i] > min_parts[i]:
                meets_requirement = True
                break
            elif current_parts[i] < min_parts[i]:
                break
            elif i == 2:  # All parts are equal
                meets_requirement = True
        
        if meets_requirement:
            status = ValidationStatus.PASSED
            message = f"Python version {current_version} meets the minimum requirement of {min_version}"
        else:
            status = ValidationStatus.FAILED
            message = f"Python version {current_version} does not meet the minimum requirement of {min_version}"
        
        return self._create_validation_result(
            name="python_version",
            category="System",
            status=status,
            message=message,
            details={
                "current_version": current_version,
                "min_required": min_version,
                "executable": sys.executable,
                "implementation": platform.python_implementation()
            }
        )
    
    def _check_required_packages(self) -> ValidationResult:
        """Check required Python packages"""
        required_packages = [
            "numpy", "pandas", "requests", "cryptography", "psutil",
            "torch", "sqlalchemy", "fastapi", "uvicorn", "schedule"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                installed_packages[package] = version
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            status = ValidationStatus.PASSED
            message = f"All required packages are installed"
        else:
            status = ValidationStatus.FAILED
            message = f"Missing required packages: {', '.join(missing_packages)}"
        
        return self._create_validation_result(
            name="required_packages",
            category="System",
            status=status,
            message=message,
            details={
                "installed_packages": installed_packages,
                "missing_packages": missing_packages
            }
        )
    
    def _check_network_connectivity(self) -> ValidationResult:
        """Check network connectivity"""
        hosts_to_check = [
            "www.google.com",
            "api.openai.com",
            "github.com",
            "pypi.org"
        ]
        
        results = {}
        all_reachable = True
        
        for host in hosts_to_check:
            try:
                # Try to resolve the hostname
                socket.gethostbyname(host)
                
                # Try to connect to port 443 (HTTPS)
                with socket.create_connection((host, 443), timeout=5) as conn:
                    pass
                
                results[host] = True
            except (socket.gaierror, socket.timeout, ConnectionRefusedError) as e:
                results[host] = False
                all_reachable = False
        
        if all_reachable:
            status = ValidationStatus.PASSED
            message = "All required hosts are reachable"
        else:
            unreachable = [host for host, reachable in results.items() if not reachable]
            status = ValidationStatus.WARNING
            message = f"Some hosts are unreachable: {', '.join(unreachable)}"
        
        return self._create_validation_result(
            name="network_connectivity",
            category="System",
            status=status,
            message=message,
            details={
                "host_results": results
            }
        )
    
    def _check_file_permissions(self) -> ValidationResult:
        """Check file permissions"""
        install_dir = self.config.install_dir or os.getcwd()
        
        # Directories to check
        dirs_to_check = [
            install_dir,
            os.path.join(install_dir, "data") if self.config.data_dir is None else self.config.data_dir,
            os.path.join(install_dir, "config") if self.config.config_dir is None else self.config.config_dir,
            os.path.join(install_dir, "logs"),
            os.path.join(install_dir, "wallets")
        ]
        
        results = {}
        all_writable = True
        
        for directory in dirs_to_check:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    results[directory] = True
                except PermissionError:
                    results[directory] = False
                    all_writable = False
            else:
                # Check if we can write to the directory
                test_file = os.path.join(directory, f"permission_test_{uuid.uuid4().hex}")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    results[directory] = True
                except (PermissionError, IOError):
                    results[directory] = False
                    all_writable = False
        
        if all_writable:
            status = ValidationStatus.PASSED
            message = "All required directories have write permissions"
        else:
            unwritable = [dir for dir, writable in results.items() if not writable]
            status = ValidationStatus.FAILED
            message = f"Missing write permissions for: {', '.join(unwritable)}"
        
        return self._create_validation_result(
            name="file_permissions",
            category="System",
            status=status,
            message=message,
            details={
                "directory_results": results
            }
        )
    
    def _check_os_compatibility(self) -> ValidationResult:
        """Check OS compatibility"""
        os_name = platform.system()
        os_version = platform.version()
        os_release = platform.release()
        
        # Check if OS is supported
        if os_name == "Windows":
            # Check Windows version
            if int(platform.win32_ver()[1].split('.')[0]) >= 10:
                status = ValidationStatus.PASSED
                message = f"Windows {os_release} is supported"
            else:
                status = ValidationStatus.WARNING
                message = f"Windows {os_release} may have limited support"
        elif os_name == "Linux":
            # All modern Linux distributions should be fine
            status = ValidationStatus.PASSED
            message = f"Linux {os_release} is supported"
        elif os_name == "Darwin":  # macOS
            # Check macOS version
            mac_ver = platform.mac_ver()[0]
            if int(mac_ver.split('.')[0]) >= 10 and int(mac_ver.split('.')[1]) >= 15:
                status = ValidationStatus.PASSED
                message = f"macOS {mac_ver} is supported"
            else:
                status = ValidationStatus.WARNING
                message = f"macOS {mac_ver} may have limited support"
        else:
            status = ValidationStatus.WARNING
            message = f"Operating system {os_name} has not been tested for compatibility"
        
        return self._create_validation_result(
            name="os_compatibility",
            category="System",
            status=status,
            message=message,
            details={
                "os_name": os_name,
                "os_version": os_version,
                "os_release": os_release,
                "platform": platform.platform()
            }
        )
    
    def _check_gpu_availability(self) -> ValidationResult:
        """Check GPU availability"""
        # Try to detect CUDA availability
        cuda_available = False
        gpu_info = {}
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_info["count"] = gpu_count
                gpu_info["devices"] = []
                
                for i in range(gpu_count):
                    device_name = torch.cuda.get_device_name(i)
                    device_capability = torch.cuda.get_device_capability(i)
                    
                    gpu_info["devices"].append({
                        "index": i,
                        "name": device_name,
                        "compute_capability": f"{device_capability[0]}.{device_capability[1]}"
                    })
                
                status = ValidationStatus.PASSED
                message = f"Found {gpu_count} CUDA-compatible GPU(s)"
            else:
                status = ValidationStatus.WARNING
                message = "No CUDA-compatible GPU detected"
        except ImportError:
            status = ValidationStatus.WARNING
            message = "Could not check GPU availability (torch not installed)"
        except Exception as e:
            status = ValidationStatus.WARNING
            message = f"Error detecting GPU: {str(e)}"
        
        return self._create_validation_result(
            name="gpu_availability",
            category="System",
            status=status,
            message=message,
            details={
                "cuda_available": cuda_available,
                "gpu_info": gpu_info
            }
        )
    
    def _check_system_time(self) -> ValidationResult:
        """Check system time accuracy"""
        # Try to get time from an NTP server
        ntp_time = None
        system_time = time.time()
        
        try:
            # Simple HTTP request to get approximate internet time
            response = requests.get("http://worldtimeapi.org/api/ip", timeout=5)
            if response.status_code == 200:
                data = response.json()
                ntp_time = data.get("unixtime")
        except Exception:
            pass
        
        if ntp_time is not None:
            # Calculate time difference
            time_diff = abs(system_time - ntp_time)
            
            if time_diff < 60:  # Less than 1 minute difference
                status = ValidationStatus.PASSED
                message = f"System time is accurate (within {time_diff:.2f} seconds of internet time)"
            elif time_diff < 300:  # Less than 5 minutes difference
                status = ValidationStatus.WARNING
                message = f"System time differs from internet time by {time_diff:.2f} seconds"
            else:
                status = ValidationStatus.FAILED
                message = f"System time is significantly off by {time_diff:.2f} seconds"
        else:
            # Could not get internet time
            status = ValidationStatus.WARNING
            message = "Could not verify system time accuracy"
        
        return self._create_validation_result(
            name="system_time",
            category="System",
            status=status,
            message=message,
            details={
                "system_time": datetime.datetime.fromtimestamp(system_time).isoformat(),
                "ntp_time": datetime.datetime.fromtimestamp(ntp_time).isoformat() if ntp_time else None,
                "time_difference_seconds": round(time_diff, 2) if ntp_time else None
            }
        )

class ConfigValidator(BaseValidator):
    """Validator for configuration files"""
    
    def validate(self) -> CategoryResult:
        """
        Validate configuration files.
        
        Returns:
            CategoryResult with validation results
        """
        start_time = time.time()
        
        category_result = CategoryResult(name="Configuration Validation")
        
        self.logger.info("Validating configuration files")
        
        # Define validation checks
        validations = [
            self._check_main_config,
            self._check_module_configs,
            self._check_agent_config,
            self._check_wallet_config,
            self._check_security_config,
            self._check_api_config,
            self._check_database_config,
            self._check_backup_config,
            self._check_monitoring_config,
            self._check_logging_config
        ]
        
        # Run each validation check
        for validation_func in validations:
            try:
                validation_name = validation_func.__name__.replace("_check_", "")
                self.logger.info(f"Checking {validation_name}")
                
                start = time.time()
                result = validation_func()
                duration = time.time() - start
                
                result.duration = duration
                category_result.results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error during {validation_func.__name__}: {e}")
                
                # Create error result
                error_result = self._create_validation_result(
                    name=validation_func.__name__.replace("_check_", ""),
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Error during validation: {str(e)}",
                    duration=0.0
                )
                
                category_result.results.append(error_result)
        
        # Calculate total duration
        end_time = time.time()
        category_result.duration = end_time - start_time
        category_result.end_time = datetime.datetime.now().isoformat()
        
        self.logger.info(f"Configuration validation completed")
        
        return category_result
    
    def _check_main_config(self) -> ValidationResult:
        """Check main system configuration"""
        config_dir = self.config.config_dir or os.path.join(self.config.install_dir or os.getcwd(), "config")
        config_path = os.path.join(config_dir, "system.ini")
        
        if not os.path.exists(config_path):
            return self._create_validation_result(
                name="main_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Main configuration file not found at {config_path}",
                details={"path": config_path}
            )
        
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Check required sections
            required_sections = ["System", "Paths", "Performance", "Security"]
            missing_sections = [section for section in required_sections if section not in config.sections()]
            
            if missing_sections:
                return self._create_validation_result(
                    name="main_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Main configuration missing required sections: {', '.join(missing_sections)}",
                    details={
                        "path": config_path,
                        "missing_sections": missing_sections,
                        "available_sections": config.sections()
                    }
                )
            
            # Check required settings in System section
            system_section = config["System"]
            required_system_settings = ["Name", "Version", "InstallDir"]
            missing_settings = [setting for setting in required_system_settings if setting not in system_section]
            
            if missing_settings:
                return self._create_validation_result(
                    name="main_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Main configuration missing required System settings: {', '.join(missing_settings)}",
                    details={
                        "path": config_path,
                        "missing_settings": missing_settings,
                        "available_settings": list(system_section.keys())
                    }
                )
            
            # Validate paths
            paths_section = config["Paths"]
            invalid_paths = []
            
            for key, path in paths_section.items():
                if not os.path.exists(path):
                    invalid_paths.append(f"{key}: {path}")
            
            if invalid_paths:
                return self._create_validation_result(
                    name="main_config",
                    category="Configuration",
                    status=ValidationStatus.WARNING,
                    message=f"Some configured paths do not exist: {', '.join(invalid_paths)}",
                    details={
                        "path": config_path,
                        "invalid_paths": invalid_paths
                    }
                )
            
            # All checks passed
            return self._create_validation_result(
                name="main_config",
                category="Configuration",
                status=ValidationStatus.PASSED,
                message=f"Main configuration validated successfully",
                details={
                    "path": config_path,
                    "sections": config.sections(),
                    "system_name": system_section.get("Name", ""),
                    "system_version": system_section.get("Version", "")
                }
            )
            
        except Exception as e:
            return self._create_validation_result(
                name="main_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Error parsing main configuration: {str(e)}",
                details={"path": config_path, "error": str(e)}
            )
    
    def _check_module_configs(self) -> ValidationResult:
        """Check module configuration files"""
        config_dir = self.config.config_dir or os.path.join(self.config.install_dir or os.getcwd(), "config")
        
        # Get list of module config files
        module_configs = []
        for file in os.listdir(config_dir):
            if file.endswith(".ini") and file != "system.ini" and file != "agents.ini" and file != "wallets.ini":
                module_configs.append(os.path.join(config_dir, file))
        
        if not module_configs:
            return self._create_validation_result(
                name="module_configs",
                category="Configuration",
                status=ValidationStatus.WARNING,
                message=f"No module configuration files found in {config_dir}",
                details={"config_dir": config_dir}
            )
        
        # Validate each module config
        valid_configs = []
        invalid_configs = []
        
        for config_path in module_configs:
            try:
                config = configparser.ConfigParser()
                config.read(config_path)
                
                # Check required sections
                required_sections = ["Module", "Strategies", "Performance"]
                missing_sections = [section for section in required_sections if section not in config.sections()]
                
                if missing_sections:
                    invalid_configs.append({
                        "path": config_path,
                        "issue": f"Missing sections: {', '.join(missing_sections)}"
                    })
                    continue
                
                # Check required settings in Module section
                module_section = config["Module"]
                required_module_settings = ["Name", "Description", "Enabled"]
                missing_settings = [setting for setting in required_module_settings if setting not in module_section]
                
                if missing_settings:
                    invalid_configs.append({
                        "path": config_path,
                        "issue": f"Missing Module settings: {', '.join(missing_settings)}"
                    })
                    continue
                
                # All checks passed for this config
                valid_configs.append({
                    "path": config_path,
                    "name": module_section.get("Name", ""),
                    "enabled": module_section.getboolean("Enabled", False)
                })
                
            except Exception as e:
                invalid_configs.append({
                    "path": config_path,
                    "issue": f"Error parsing configuration: {str(e)}"
                })
        
        if not invalid_configs:
            return self._create_validation_result(
                name="module_configs",
                category="Configuration",
                status=ValidationStatus.PASSED,
                message=f"All {len(valid_configs)} module configurations validated successfully",
                details={
                    "valid_configs": valid_configs
                }
            )
        else:
            return self._create_validation_result(
                name="module_configs",
                category="Configuration",
                status=ValidationStatus.WARNING,
                message=f"{len(invalid_configs)} of {len(module_configs)} module configurations have issues",
                details={
                    "valid_configs": valid_configs,
                    "invalid_configs": invalid_configs
                }
            )
    
    def _check_agent_config(self) -> ValidationResult:
        """Check agent configuration"""
        config_dir = self.config.config_dir or os.path.join(self.config.install_dir or os.getcwd(), "config")
        config_path = os.path.join(config_dir, "agents.ini")
        
        if not os.path.exists(config_path):
            return self._create_validation_result(
                name="agent_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Agent configuration file not found at {config_path}",
                details={"path": config_path}
            )
        
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Check required sections
            required_sections = ["Agents", "Behavior", "Learning"]
            missing_sections = [section for section in required_sections if section not in config.sections()]
            
            if missing_sections:
                return self._create_validation_result(
                    name="agent_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Agent configuration missing required sections: {', '.join(missing_sections)}",
                    details={
                        "path": config_path,
                        "missing_sections": missing_sections,
                        "available_sections": config.sections()
                    }
                )
            
            # Check required settings in Agents section
            agents_section = config["Agents"]
            required_agent_settings = ["TotalCount", "ManagerCount", "WorkerCount"]
            missing_settings = [setting for setting in required_agent_settings if setting not in agents_section]
            
            if missing_settings:
                return self._create_validation_result(
                    name="agent_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Agent configuration missing required Agents settings: {', '.join(missing_settings)}",
                    details={
                        "path": config_path,
                        "missing_settings": missing_settings,
                        "available_settings": list(agents_section.keys())
                    }
                )
            
            # Check agent counts
            try:
                total_count = int(agents_section.get("TotalCount", "0"))
                manager_count = int(agents_section.get("ManagerCount", "0"))
                worker_count = int(agents_section.get("WorkerCount", "0"))
                analyst_count = int(agents_section.get("AnalystCount", "0"))
                specialist_count = int(agents_section.get("SpecialistCount", "0"))
                
                sum_counts = manager_count + worker_count + analyst_count + specialist_count
                
                if sum_counts != total_count:
                    return self._create_validation_result(
                        name="agent_config",
                        category="Configuration",
                        status=ValidationStatus.WARNING,
                        message=f"Agent counts don't add up: Total={total_count}, Sum={sum_counts}",
                        details={
                            "path": config_path,
                            "total_count": total_count,
                            "manager_count": manager_count,
                            "worker_count": worker_count,
                            "analyst_count": analyst_count,
                            "specialist_count": specialist_count,
                            "sum_counts": sum_counts
                        }
                    )
            except ValueError:
                return self._create_validation_result(
                    name="agent_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Agent configuration contains invalid numeric values",
                    details={
                        "path": config_path,
                        "TotalCount": agents_section.get("TotalCount", ""),
                        "ManagerCount": agents_section.get("ManagerCount", ""),
                        "WorkerCount": agents_section.get("WorkerCount", "")
                    }
                )
            
            # All checks passed
            return self._create_validation_result(
                name="agent_config",
                category="Configuration",
                status=ValidationStatus.PASSED,
                message=f"Agent configuration validated successfully",
                details={
                    "path": config_path,
                    "total_agents": total_count,
                    "manager_count": manager_count,
                    "worker_count": worker_count,
                    "analyst_count": analyst_count,
                    "specialist_count": specialist_count
                }
            )
            
        except Exception as e:
            return self._create_validation_result(
                name="agent_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Error parsing agent configuration: {str(e)}",
                details={"path": config_path, "error": str(e)}
            )
    
    def _check_wallet_config(self) -> ValidationResult:
        """Check wallet configuration"""
        config_dir = self.config.config_dir or os.path.join(self.config.install_dir or os.getcwd(), "config")
        config_path = os.path.join(config_dir, "wallets.ini")
        
        if not os.path.exists(config_path):
            return self._create_validation_result(
                name="wallet_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Wallet configuration file not found at {config_path}",
                details={"path": config_path}
            )
        
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Check required sections
            required_sections = ["Wallets", "Security"]
            missing_sections = [section for section in required_sections if section not in config.sections()]
            
            if missing_sections:
                return self._create_validation_result(
                    name="wallet_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Wallet configuration missing required sections: {', '.join(missing_sections)}",
                    details={
                        "path": config_path,
                        "missing_sections": missing_sections,
                        "available_sections": config.sections()
                    }
                )
            
            # Check required settings in Wallets section
            wallets_section = config["Wallets"]
            required_wallet_settings = ["EncryptionEnabled", "AutoBackup"]
            missing_settings = [setting for setting in required_wallet_settings if setting not in wallets_section]
            
            if missing_settings:
                return self._create_validation_result(
                    name="wallet_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Wallet configuration missing required Wallets settings: {', '.join(missing_settings)}",
                    details={
                        "path": config_path,
                        "missing_settings": missing_settings,
                        "available_settings": list(wallets_section.keys())
                    }
                )
            
            # Check security settings
            security_section = config["Security"]
            required_security_settings = ["RequirePIN", "PINTimeout"]
            missing_settings = [setting for setting in required_security_settings if setting not in security_section]
            
            if missing_settings:
                return self._create_validation_result(
                    name="wallet_config",
                    category="Configuration",
                    status=ValidationStatus.WARNING,
                    message=f"Wallet configuration missing recommended Security settings: {', '.join(missing_settings)}",
                    details={
                        "path": config_path,
                        "missing_settings": missing_settings,
                        "available_settings": list(security_section.keys())
                    }
                )
            
            # Check encryption is enabled
            if not wallets_section.getboolean("EncryptionEnabled", False):
                return self._create_validation_result(
                    name="wallet_config",
                    category="Configuration",
                    status=ValidationStatus.WARNING,
                    message=f"Wallet encryption is disabled, which is not recommended for production",
                    details={
                        "path": config_path,
                        "EncryptionEnabled": False
                    }
                )
            
            # All checks passed
            return self._create_validation_result(
                name="wallet_config",
                category="Configuration",
                status=ValidationStatus.PASSED,
                message=f"Wallet configuration validated successfully",
                details={
                    "path": config_path,
                    "encryption_enabled": wallets_section.getboolean("EncryptionEnabled", False),
                    "auto_backup": wallets_section.getboolean("AutoBackup", False),
                    "require_pin": security_section.getboolean("RequirePIN", False)
                }
            )
            
        except Exception as e:
            return self._create_validation_result(
                name="wallet_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Error parsing wallet configuration: {str(e)}",
                details={"path": config_path, "error": str(e)}
            )
    
    def _check_security_config(self) -> ValidationResult:
        """Check security configuration"""
        config_dir = self.config.config_dir or os.path.join(self.config.install_dir or os.getcwd(), "config")
        
        # Look for security-related settings in system.ini
        system_config_path = os.path.join(config_dir, "system.ini")
        
        if not os.path.exists(system_config_path):
            return self._create_validation_result(
                name="security_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"System configuration file not found at {system_config_path}",
                details={"path": system_config_path}
            )
        
        try:
            config = configparser.ConfigParser()
            config.read(system_config_path)
            
            # Check if Security section exists
            if "Security" not in config.sections():
                return self._create_validation_result(
                    name="security_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Security section missing in system configuration",
                    details={
                        "path": system_config_path,
                        "available_sections": config.sections()
                    }
                )
            
            # Check required security settings
            security_section = config["Security"]
            required_security_settings = [
                "EncryptWallets", "AutoLockTimeout", "RequirePIN",
                "AllowRemoteAccess", "EnableFirewall"
            ]
            missing_settings = [setting for setting in required_security_settings if setting not in security_section]
            
            if missing_settings:
                return self._create_validation_result(
                    name="security_config",
                    category="Configuration",
                    status=ValidationStatus.WARNING,
                    message=f"Security configuration missing recommended settings: {', '.join(missing_settings)}",
                    details={
                        "path": system_config_path,
                        "missing_settings": missing_settings,
                        "available_settings": list(security_section.keys())
                    }
                )
            
            # Check security configuration values
            security_issues = []
            
            if not security_section.getboolean("EncryptWallets", False):
                security_issues.append("Wallet encryption is disabled")
            
            if security_section.getboolean("AllowRemoteAccess", False):
                security_issues.append("Remote access is enabled")
            
            if not security_section.getboolean("EnableFirewall", True):
                security_issues.append("Firewall is disabled")
            
            if not security_section.getboolean("RequirePIN", True):
                security_issues.append("PIN authentication is disabled")
            
            try:
                auto_lock_timeout = int(security_section.get("AutoLockTimeout", "0"))
                if auto_lock_timeout == 0 or auto_lock_timeout > 900:  # More than 15 minutes
                    security_issues.append(f"Auto-lock timeout is too long: {auto_lock_timeout} seconds")
            except ValueError:
                security_issues.append("Invalid auto-lock timeout value")
            
            if security_issues:
                return self._create_validation_result(
                    name="security_config",
                    category="Configuration",
                    status=ValidationStatus.WARNING,
                    message=f"Security configuration has potential issues: {', '.join(security_issues)}",
                    details={
                        "path": system_config_path,
                        "security_issues": security_issues,
                        "security_settings": {k: v for k, v in security_section.items()}
                    }
                )
            
            # All checks passed
            return self._create_validation_result(
                name="security_config",
                category="Configuration",
                status=ValidationStatus.PASSED,
                message=f"Security configuration validated successfully",
                details={
                    "path": system_config_path,
                    "security_settings": {k: v for k, v in security_section.items()}
                }
            )
            
        except Exception as e:
            return self._create_validation_result(
                name="security_config",
                category="Configuration",
                status=ValidationStatus.FAILED,
                message=f"Error parsing security configuration: {str(e)}",
                details={"path": system_config_path, "error": str(e)}
            )
    
    def _check_api_config(self) -> ValidationResult:
        """Check API configuration"""
        config_dir = self.config.config_dir or os.path.join(self.config.install_dir or os.getcwd(), "config")
        
        # Look for API configuration file
        api_config_path = os.path.join(config_dir, "api.ini")
        
        if not os.path.exists(api_config_path):
            # Try looking for API settings in system.ini
            system_config_path = os.path.join(config_dir, "system.ini")
            
            if not os.path.exists(system_config_path):
                return self._create_validation_result(
                    name="api_config",
                    category="Configuration",
                    status=ValidationStatus.FAILED,
                    message=f"API configuration file not found at {api_config_path} and system.ini not found",
                    details={
                        "api_path": api_config_path,
                        "system_path": system_config_path
                    }
                )
            
            try:
                config = configparser.ConfigParser()
                config.read(system_config_path)
                
                # Check if API section exists
                if "API" not in config.sections():
                    return self._create_validation_result(
                        name="api_config",
                        category="Configuration",
                        status=ValidationStatus.WARNING,
                        message=f"API section missing in system configuration",
                        details={
                            "path": system_config_path,
                            "available_sections": config.sections()
                        }
                    )
                
                # Check API settings
                api_section = config["API"]
                required_api_settings = ["Enabled", "Host", "Port", "RequireAuth"]
                missing_settings = [setting for setting in required_api_settings if setting not in api_section]
                
                if missing_settings:
                    return self._create_validation_result(
                        name="api_config",
                        category="Configuration",
                        status=ValidationStatus.WARNING,
                        message=f"API configuration missing recommended settings: {', '.join(missing_settings)}",
                        details={
                            "path": system_config_path,
                            "missing_settings": missing_settings,
                            "available_settings": list(api_section.keys())
                        }
                    )
                
                # Check API security
                if not api_section.getboolean("RequireAuth", True):
                    return self._create_validation_result(
                        name="api_config",
                        category="Configuration",
                        status=ValidationStatus.WARNING,
                        message=f"API authentication is disabled, which is not recommended for production",
                        details={
                            "path": system_config_path,
                            "RequireAuth": False
                        }
                    )
                
                # All checks passed
                return self._create_validation_result(
                    name="api_config",
                    category="Configuration",
                    status=ValidationStatus.PASSED,
                    message=f"API configuration validated successfully",
                    details