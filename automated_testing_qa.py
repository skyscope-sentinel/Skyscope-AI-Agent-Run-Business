import os
import sys
import json
import time
import uuid
import logging
import datetime
import threading
import traceback
import unittest
import inspect
import coverage
import pytest
import requests
import subprocess
import re
import random
import asyncio
import concurrent.futures
import queue
import signal
import tempfile
import shutil
import socket
import psutil
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Generator, Type
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import multiprocessing as mp
from functools import wraps, partial

# Test frameworks and tools
import pytest
import hypothesis
from hypothesis import given, strategies as st
import locust
from locust import HttpUser, task, between
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Data processing and analysis
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Security testing
try:
    import bandit
    import safety
    import owasp_zap_api_python_client
    from owasp_zap_api_python_client.zap import ZAPv2
    SECURITY_TOOLS_AVAILABLE = True
except ImportError:
    SECURITY_TOOLS_AVAILABLE = False

# Performance testing
try:
    import locust
    import pyinstrument
    from pyinstrument import Profiler
    PERF_TOOLS_AVAILABLE = True
except ImportError:
    PERF_TOOLS_AVAILABLE = False

# Import internal modules
try:
    from agent_manager import AgentManager
    from business_manager import BusinessManager
    from crypto_manager import CryptoManager
    from database_manager import DatabaseManager
    from performance_monitor import PerformanceMonitor
    from live_thinking_rag_system import LiveThinkingRAGSystem
    from advanced_business_operations import AnalyticsEngine
    from enhanced_security_compliance import AuditLogger, EncryptionManager
    from realtime_analytics_dashboard import DataProcessor, VisualizationEngine
    INTERNAL_MODULES_AVAILABLE = True
except ImportError:
    INTERNAL_MODULES_AVAILABLE = False
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automated_testing_qa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automated_testing_qa")

# Constants
CONFIG_DIR = Path("config")
TEST_DIR = Path("tests")
REPORTS_DIR = Path("reports")
TEST_DATA_DIR = Path("test_data")
COVERAGE_DIR = Path("coverage")
BENCHMARK_DIR = Path("benchmarks")
FIXTURES_DIR = Path("fixtures")
SNAPSHOTS_DIR = Path("snapshots")
TEMP_DIR = Path("temp")
LOGS_DIR = Path("logs")

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "testing_config.json"

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    PERFORMANCE = "performance"
    SECURITY = "security"
    LOAD = "load"
    STRESS = "stress"
    CHAOS = "chaos"
    REGRESSION = "regression"
    SMOKE = "smoke"
    SANITY = "sanity"
    ACCEPTANCE = "acceptance"
    CUSTOM = "custom"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    FLAKY = "flaky"
    TIMEOUT = "timeout"

class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TestEnvironment(Enum):
    """Test execution environments."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CI = "ci"

class TestingMode(Enum):
    """Testing modes."""
    NORMAL = "normal"
    QUICK = "quick"
    THOROUGH = "thorough"
    CONTINUOUS = "continuous"
    NIGHTLY = "nightly"
    WEEKLY = "weekly"

class AgentTestMode(Enum):
    """Agent testing modes."""
    SINGLE = "single"
    SUBSET = "subset"
    FULL = "full"
    SIMULATED = "simulated"

class FixtureScope(Enum):
    """Fixture scopes."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SESSION = "session"

@dataclass
class TestConfig:
    """Configuration for the testing system."""
    enabled_test_types: List[TestType] = field(default_factory=lambda: list(TestType))
    test_environment: TestEnvironment = TestEnvironment.LOCAL
    testing_mode: TestingMode = TestingMode.NORMAL
    agent_test_mode: AgentTestMode = AgentTestMode.SUBSET
    agent_subset_size: int = 100  # Number of agents to test in SUBSET mode
    parallel_tests: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    retry_count: int = 2
    capture_screenshots: bool = True
    record_video: bool = False
    enable_coverage: bool = True
    coverage_threshold: float = 80.0
    enable_mutation_testing: bool = False
    enable_property_testing: bool = True
    property_test_iterations: int = 100
    enable_performance_profiling: bool = True
    enable_security_scanning: bool = True
    enable_self_healing: bool = True
    healing_strategies: List[str] = field(default_factory=lambda: ["retry", "parameter_adjustment", "timeout_extension"])
    enable_continuous_monitoring: bool = True
    monitoring_interval_seconds: int = 60
    report_formats: List[str] = field(default_factory=lambda: ["html", "json", "xml"])
    notification_channels: List[str] = field(default_factory=lambda: ["console", "log"])
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled_test_types": [tt.value for tt in self.enabled_test_types],
            "test_environment": self.test_environment.value,
            "testing_mode": self.testing_mode.value,
            "agent_test_mode": self.agent_test_mode.value,
            "agent_subset_size": self.agent_subset_size,
            "parallel_tests": self.parallel_tests,
            "max_workers": self.max_workers,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "capture_screenshots": self.capture_screenshots,
            "record_video": self.record_video,
            "enable_coverage": self.enable_coverage,
            "coverage_threshold": self.coverage_threshold,
            "enable_mutation_testing": self.enable_mutation_testing,
            "enable_property_testing": self.enable_property_testing,
            "property_test_iterations": self.property_test_iterations,
            "enable_performance_profiling": self.enable_performance_profiling,
            "enable_security_scanning": self.enable_security_scanning,
            "enable_self_healing": self.enable_self_healing,
            "healing_strategies": self.healing_strategies,
            "enable_continuous_monitoring": self.enable_continuous_monitoring,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "report_formats": self.report_formats,
            "notification_channels": self.notification_channels,
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestConfig':
        """Create from dictionary."""
        return cls(
            enabled_test_types=[TestType(tt) for tt in data.get("enabled_test_types", [tt.value for tt in TestType])],
            test_environment=TestEnvironment(data.get("test_environment", TestEnvironment.LOCAL.value)),
            testing_mode=TestingMode(data.get("testing_mode", TestingMode.NORMAL.value)),
            agent_test_mode=AgentTestMode(data.get("agent_test_mode", AgentTestMode.SUBSET.value)),
            agent_subset_size=data.get("agent_subset_size", 100),
            parallel_tests=data.get("parallel_tests", True),
            max_workers=data.get("max_workers", 4),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 2),
            capture_screenshots=data.get("capture_screenshots", True),
            record_video=data.get("record_video", False),
            enable_coverage=data.get("enable_coverage", True),
            coverage_threshold=data.get("coverage_threshold", 80.0),
            enable_mutation_testing=data.get("enable_mutation_testing", False),
            enable_property_testing=data.get("enable_property_testing", True),
            property_test_iterations=data.get("property_test_iterations", 100),
            enable_performance_profiling=data.get("enable_performance_profiling", True),
            enable_security_scanning=data.get("enable_security_scanning", True),
            enable_self_healing=data.get("enable_self_healing", True),
            healing_strategies=data.get("healing_strategies", ["retry", "parameter_adjustment", "timeout_extension"]),
            enable_continuous_monitoring=data.get("enable_continuous_monitoring", True),
            monitoring_interval_seconds=data.get("monitoring_interval_seconds", 60),
            report_formats=data.get("report_formats", ["html", "json", "xml"]),
            notification_channels=data.get("notification_channels", ["console", "log"]),
            custom_settings=data.get("custom_settings", {})
        )
    
    def save(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Test configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving test configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = DEFAULT_CONFIG_PATH) -> 'TestConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Test configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading test configuration: {e}")
            return cls()

@dataclass
class TestCase:
    """Base test case definition."""
    id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority = TestPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 60
    retry_count: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    test_function: Optional[Callable] = None
    status: TestStatus = TestStatus.PENDING
    execution_time: float = 0.0
    last_run: Optional[datetime.datetime] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "priority": self.priority.value,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "parameters": self.parameters,
            "expected_results": self.expected_results,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "error_message": self.error_message,
            "custom_properties": self.custom_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create from dictionary."""
        last_run = datetime.datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            test_type=TestType(data["test_type"]),
            priority=TestPriority(data.get("priority", TestPriority.MEDIUM.value)),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            timeout_seconds=data.get("timeout_seconds", 60),
            retry_count=data.get("retry_count", 0),
            parameters=data.get("parameters", {}),
            expected_results=data.get("expected_results", {}),
            status=TestStatus(data.get("status", TestStatus.PENDING.value)),
            execution_time=data.get("execution_time", 0.0),
            last_run=last_run,
            error_message=data.get("error_message"),
            custom_properties=data.get("custom_properties", {})
        )

@dataclass
class TestSuite:
    """Collection of test cases."""
    id: str
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    execution_time: float = 0.0
    last_run: Optional[datetime.datetime] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "tags": self.tags,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "custom_properties": self.custom_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestSuite':
        """Create from dictionary."""
        last_run = datetime.datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            test_cases=[TestCase.from_dict(tc) for tc in data.get("test_cases", [])],
            tags=data.get("tags", []),
            status=TestStatus(data.get("status", TestStatus.PENDING.value)),
            execution_time=data.get("execution_time", 0.0),
            last_run=last_run,
            custom_properties=data.get("custom_properties", {})
        )

@dataclass
class TestResult:
    """Result of a test execution."""
    test_id: str
    test_name: str
    status: TestStatus
    start_time: datetime.datetime
    end_time: datetime.datetime
    execution_time: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    video_path: Optional[str] = None
    logs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "screenshots": self.screenshots,
            "video_path": self.video_path,
            "logs": self.logs,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """Create from dictionary."""
        start_time = datetime.datetime.fromisoformat(data["start_time"])
        end_time = datetime.datetime.fromisoformat(data["end_time"])
        
        return cls(
            test_id=data["test_id"],
            test_name=data["test_name"],
            status=TestStatus(data["status"]),
            start_time=start_time,
            end_time=end_time,
            execution_time=data["execution_time"],
            error_message=data.get("error_message"),
            stack_trace=data.get("stack_trace"),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            screenshots=data.get("screenshots", []),
            video_path=data.get("video_path"),
            logs=data.get("logs", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            custom_data=data.get("custom_data", {})
        )

@dataclass
class TestReport:
    """Report of test execution results."""
    id: str
    name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    execution_time: float
    test_results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "execution_time": self.execution_time,
            "test_results": [tr.to_dict() for tr in self.test_results],
            "summary": self.summary,
            "coverage": self.coverage,
            "environment": self.environment,
            "metrics": self.metrics,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestReport':
        """Create from dictionary."""
        start_time = datetime.datetime.fromisoformat(data["start_time"])
        end_time = datetime.datetime.fromisoformat(data["end_time"])
        
        return cls(
            id=data["id"],
            name=data["name"],
            start_time=start_time,
            end_time=end_time,
            execution_time=data["execution_time"],
            test_results=[TestResult.from_dict(tr) for tr in data.get("test_results", [])],
            summary=data.get("summary", {}),
            coverage=data.get("coverage", {}),
            environment=data.get("environment", {}),
            metrics=data.get("metrics", {}),
            custom_data=data.get("custom_data", {})
        )
    
    def save(self, filepath: Optional[Path] = None, format: str = "json") -> Path:
        """Save report to file."""
        if filepath is None:
            timestamp = self.end_time.strftime("%Y%m%d_%H%M%S")
            filepath = REPORTS_DIR / f"test_report_{timestamp}.{format}"
        
        try:
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.to_dict(), f, indent=2)
            elif format == "html":
                html_content = self._generate_html_report()
                with open(filepath, 'w') as f:
                    f.write(html_content)
            elif format == "xml":
                xml_content = self._generate_xml_report()
                with open(filepath, 'w') as f:
                    f.write(xml_content)
            else:
                raise ValueError(f"Unsupported report format: {format}")
            
            logger.info(f"Test report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving test report: {e}")
            raise
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        # Calculate summary statistics
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        error = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.name} - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ display: flex; margin-bottom: 20px; }}
        .summary-item {{ margin-right: 20px; padding: 10px; border-radius: 5px; }}
        .total {{ background-color: #f0f0f0; }}
        .passed {{ background-color: #dff0d8; }}
        .failed {{ background-color: #f2dede; }}
        .error {{ background-color: #fcf8e3; }}
        .skipped {{ background-color: #d9edf7; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr.passed {{ background-color: #dff0d8; }}
        tr.failed {{ background-color: #f2dede; }}
        tr.error {{ background-color: #fcf8e3; }}
        tr.skipped {{ background-color: #d9edf7; }}
        .details {{ margin-top: 5px; padding: 10px; border: 1px solid #ddd; display: none; }}
        .toggle-details {{ cursor: pointer; color: blue; text-decoration: underline; }}
    </style>
    <script>
        function toggleDetails(id) {{
            var details = document.getElementById(id);
            if (details.style.display === "none") {{
                details.style.display = "block";
            }} else {{
                details.style.display = "none";
            }}
        }}
    </script>
</head>
<body>
    <h1>{self.name} - Test Report</h1>
    <p>Generated on: {self.end_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>Execution time: {self.execution_time:.2f} seconds</p>
    
    <h2>Summary</h2>
    <div class="summary">
        <div class="summary-item total">Total: {total}</div>
        <div class="summary-item passed">Passed: {passed}</div>
        <div class="summary-item failed">Failed: {failed}</div>
        <div class="summary-item error">Error: {error}</div>
        <div class="summary-item skipped">Skipped: {skipped}</div>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Execution Time (s)</th>
            <th>Details</th>
        </tr>
"""
        
        # Add test results
        for i, result in enumerate(self.test_results):
            details_id = f"details_{i}"
            
            html += f"""
        <tr class="{result.status.value}">
            <td>{result.test_name}</td>
            <td>{result.status.value}</td>
            <td>{result.execution_time:.2f}</td>
            <td><span class="toggle-details" onclick="toggleDetails('{details_id}')">Show Details</span></td>
        </tr>
        <tr>
            <td colspan="4">
                <div id="{details_id}" class="details">
"""
            
            # Add error message and stack trace if available
            if result.error_message:
                html += f"<p><strong>Error:</strong> {result.error_message}</p>"
            
            if result.stack_trace:
                html += f"<pre>{result.stack_trace}</pre>"
            
            # Add screenshots if available
            if result.screenshots:
                html += "<p><strong>Screenshots:</strong></p>"
                for screenshot in result.screenshots:
                    html += f'<p><img src="{screenshot}" width="600"></p>'
            
            html += """
                </div>
            </td>
        </tr>"""
        
        # Add coverage information if available
        if self.coverage:
            html += """
    </table>
    
    <h2>Coverage</h2>
    <table>
        <tr>
            <th>Module</th>
            <th>Coverage (%)</th>
        </tr>
"""
            
            for module, cov in self.coverage.items():
                html += f"""
        <tr>
            <td>{module}</td>
            <td>{cov:.2f}%</td>
        </tr>"""
        
        # Close HTML
        html += """
    </table>
</body>
</html>
"""
        
        return html
    
    def _generate_xml_report(self) -> str:
        """Generate XML report (JUnit format)."""
        # Calculate summary statistics
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        error = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
        
        # Generate XML
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="{self.name}" tests="{total}" failures="{failed}" errors="{error}" skipped="{skipped}" time="{self.execution_time}">
    <testsuite name="{self.name}" tests="{total}" failures="{failed}" errors="{error}" skipped="{skipped}" time="{self.execution_time}">
"""
        
        # Add test cases
        for result in self.test_results:
            xml += f'        <testcase name="{result.test_name}" classname="{result.test_id}" time="{result.execution_time}">\n'
            
            if result.status == TestStatus.FAILED:
                xml += f'            <failure message="{result.error_message or "Test failed"}" type="AssertionError">\n'
                if result.stack_trace:
                    xml += f'                <![CDATA[{result.stack_trace}]]>\n'
                xml += '            </failure>\n'
            elif result.status == TestStatus.ERROR:
                xml += f'            <error message="{result.error_message or "Test error"}" type="Error">\n'
                if result.stack_trace:
                    xml += f'                <![CDATA[{result.stack_trace}]]>\n'
                xml += '            </error>\n'
            elif result.status == TestStatus.SKIPPED:
                xml += '            <skipped/>\n'
            
            xml += '        </testcase>\n'
        
        # Close XML
        xml += """    </testsuite>
</testsuites>
"""
        
        return xml

class TestFixture:
    """Test fixture for setting up and tearing down test environments."""
    
    def __init__(self, name: str, scope: FixtureScope = FixtureScope.FUNCTION):
        """Initialize the test fixture."""
        self.name = name
        self.scope = scope
        self.setup_function = None
        self.teardown_function = None
        self.parameters = {}
    
    def setup(self, func: Callable) -> Callable:
        """Decorator to set the setup function."""
        self.setup_function = func
        return func
    
    def teardown(self, func: Callable) -> Callable:
        """Decorator to set the teardown function."""
        self.teardown_function = func
        return func
    
    @contextmanager
    def __call__(self, **kwargs) -> Generator:
        """Use the fixture as a context manager."""
        # Merge parameters
        params = {**self.parameters, **kwargs}
        
        # Setup
        fixture_data = None
        if self.setup_function:
            fixture_data = self.setup_function(**params)
        
        try:
            # Yield fixture data to the test
            yield fixture_data
        finally:
            # Teardown
            if self.teardown_function:
                self.teardown_function(fixture_data, **params)

class BaseTest(ABC):
    """Base class for all test implementations."""
    
    def __init__(self, config: TestConfig = None):
        """Initialize the base test."""
        self.config = config or TestConfig.load()
        self.fixtures = {}
    
    def create_fixture(self, name: str, scope: FixtureScope = FixtureScope.FUNCTION) -> TestFixture:
        """Create a new test fixture."""
        fixture = TestFixture(name, scope)
        self.fixtures[name] = fixture
        return fixture
    
    def get_fixture(self, name: str) -> Optional[TestFixture]:
        """Get a fixture by name."""
        return self.fixtures.get(name)
    
    @abstractmethod
    def run(self) -> TestResult:
        """Run the test."""
        pass
    
    def assert_equal(self, actual: Any, expected: Any, message: str = None) -> None:
        """Assert that two values are equal."""
        assert actual == expected, message or f"Expected {expected}, got {actual}"
    
    def assert_not_equal(self, actual: Any, expected: Any, message: str = None) -> None:
        """Assert that two values are not equal."""
        assert actual != expected, message or f"Expected {actual} to be different from {expected}"
    
    def assert_true(self, condition: bool, message: str = None) -> None:
        """Assert that a condition is true."""
        assert condition, message or "Expected condition to be true"
    
    def assert_false(self, condition: bool, message: str = None) -> None:
        """Assert that a condition is false."""
        assert not condition, message or "Expected condition to be false"
    
    def assert_is_none(self, value: Any, message: str = None) -> None:
        """Assert that a value is None."""
        assert value is None, message or f"Expected None, got {value}"
    
    def assert_is_not_none(self, value: Any, message: str = None) -> None:
        """Assert that a value is not None."""
        assert value is not None, message or "Expected value to not be None"
    
    def assert_in(self, item: Any, container: Any, message: str = None) -> None:
        """Assert that an item is in a container."""
        assert item in container, message or f"Expected {item} to be in {container}"
    
    def assert_not_in(self, item: Any, container: Any, message: str = None) -> None:
        """Assert that an item is not in a container."""
        assert item not in container, message or f"Expected {item} to not be in {container}"
    
    def assert_raises(self, exception_type: Type[Exception], callable_obj: Callable, *args, **kwargs) -> None:
        """Assert that a callable raises a specific exception."""
        try:
            callable_obj(*args, **kwargs)
        except Exception as e:
            if isinstance(e, exception_type):
                return
            raise AssertionError(f"Expected {exception_type.__name__}, got {type(e).__name__}")
        
        raise AssertionError(f"Expected {exception_type.__name__}, but no exception was raised")
    
    def skip_test(self, reason: str) -> None:
        """Skip the current test."""
        raise unittest.SkipTest(reason)
    
    def fail_test(self, message: str) -> None:
        """Fail the current test."""
        raise AssertionError(message)

class UnitTest(BaseTest):
    """Unit test implementation."""
    
    def __init__(self, test_case: TestCase, config: TestConfig = None):
        """Initialize the unit test."""
        super().__init__(config)
        self.test_case = test_case
    
    def run(self) -> TestResult:
        """Run the unit test."""
        start_time = datetime.datetime.now()
        
        # Initialize test result
        result = TestResult(
            test_id=self.test_case.id,
            test_name=self.test_case.name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=start_time,  # Will be updated later
            execution_time=0.0
        )
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Run setup if available
                if self.test_case.setup_function:
                    self.test_case.setup_function()
                
                # Run the test with timeout
                with timeout(self.test_case.timeout_seconds):
                    if self.test_case.test_function:
                        self.test_case.test_function(self)
                
                # Test passed
                result.status = TestStatus.PASSED
            except unittest.SkipTest as e:
                # Test skipped
                result.status = TestStatus.SKIPPED
                result.error_message = str(e)
            except TimeoutError:
                # Test timed out
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {self.test_case.timeout_seconds} seconds"
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            finally:
                # Run teardown if available
                try:
                    if self.test_case.teardown_function:
                        self.test_case.teardown_function()
                except Exception as e:
                    # Log teardown errors but don't change test status if it passed
                    logger.error(f"Error in teardown: {e}")
                    if result.status == TestStatus.PASSED:
                        result.status = TestStatus.ERROR
                        result.error_message = f"Error in teardown: {e}"
                        result.stack_trace = traceback.format_exc()
        
        # Update test result
        end_time = datetime.datetime.now()
        result.end_time = end_time
        result.execution_time = (end_time - start_time).total_seconds()
        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        
        # Update test case
        self.test_case.status = result.status
        self.test_case.execution_time = result.execution_time
        self.test_case.last_run = end_time
        self.test_case.error_message = result.error_message
        self.test_case.stack_trace = result.stack_trace
        
        return result

class IntegrationTest(BaseTest):
    """Integration test implementation."""
    
    def __init__(self, test_case: TestCase, config: TestConfig = None):
        """Initialize the integration test."""
        super().__init__(config)
        self.test_case = test_case
        self.components = {}
    
    def setup_component(self, name: str, component: Any) -> None:
        """Set up a component for testing."""
        self.components[name] = component
    
    def get_component(self, name: str) -> Any:
        """Get a component by name."""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not found")
        return self.components[name]
    
    def run(self) -> TestResult:
        """Run the integration test."""
        start_time = datetime.datetime.now()
        
        # Initialize test result
        result = TestResult(
            test_id=self.test_case.id,
            test_name=self.test_case.name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=start_time,  # Will be updated later
            execution_time=0.0
        )
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Run setup if available
                if self.test_case.setup_function:
                    self.test_case.setup_function()
                
                # Run the test with timeout
                with timeout(self.test_case.timeout_seconds):
                    if self.test_case.test_function:
                        self.test_case.test_function(self)
                
                # Test passed
                result.status = TestStatus.PASSED
            except unittest.SkipTest as e:
                # Test skipped
                result.status = TestStatus.SKIPPED
                result.error_message = str(e)
            except TimeoutError:
                # Test timed out
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {self.test_case.timeout_seconds} seconds"
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            finally:
                # Run teardown if available
                try:
                    if self.test_case.teardown_function:
                        self.test_case.teardown_function()
                except Exception as e:
                    # Log teardown errors but don't change test status if it passed
                    logger.error(f"Error in teardown: {e}")
                    if result.status == TestStatus.PASSED:
                        result.status = TestStatus.ERROR
                        result.error_message = f"Error in teardown: {e}"
                        result.stack_trace = traceback.format_exc()
        
        # Update test result
        end_time = datetime.datetime.now()
        result.end_time = end_time
        result.execution_time = (end_time - start_time).total_seconds()
        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        
        # Update test case
        self.test_case.status = result.status
        self.test_case.execution_time = result.execution_time
        self.test_case.last_run = end_time
        self.test_case.error_message = result.error_message
        self.test_case.stack_trace = result.stack_trace
        
        return result

class EndToEndTest(BaseTest):
    """End-to-end test implementation."""
    
    def __init__(self, test_case: TestCase, config: TestConfig = None):
        """Initialize the end-to-end test."""
        super().__init__(config)
        self.test_case = test_case
        self.driver = None
        self.screenshots = []
    
    def setup_browser(self, browser_type: str = "chrome", headless: bool = True) -> None:
        """Set up a browser for testing."""
        if browser_type.lower() == "chrome":
            options = webdriver.ChromeOptions()
            if headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(options=options)
        elif browser_type.lower() == "firefox":
            options = webdriver.FirefoxOptions()
            if headless:
                options.add_argument("--headless")
            self.driver = webdriver.Firefox(options=options)
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")
        
        self.driver.implicitly_wait(10)
    
    def navigate_to(self, url: str) -> None:
        """Navigate to a URL."""
        if not self.driver:
            raise ValueError("Browser not initialized. Call setup_browser first.")
        self.driver.get(url)
    
    def find_element(self, selector: str, by: str = By.CSS_SELECTOR, timeout: int = 10) -> Any:
        """Find an element on the page."""
        if not self.driver:
            raise ValueError("Browser not initialized. Call setup_browser first.")
        
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
        except TimeoutException:
            raise ValueError(f"Element not found: {selector} (by {by})")
    
    def click(self, selector: str, by: str = By.CSS_SELECTOR) -> None:
        """Click an element."""
        element = self.find_element(selector, by)
        element.click()
    
    def type_text(self, selector: str, text: str, by: str = By.CSS_SELECTOR) -> None:
        """Type text into an element."""
        element = self.find_element(selector, by)
        element.clear()
        element.send_keys(text)
    
    def take_screenshot(self, name: str = None) -> str:
        """Take a screenshot."""
        if not self.driver:
            raise ValueError("Browser not initialized. Call setup_browser first.")
        
        if not name:
            name = f"screenshot_{len(self.screenshots) + 1}"
        
        # Create screenshots directory if it doesn't exist
        screenshots_dir = TEMP_DIR / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save screenshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = screenshots_dir / filename
        
        self.driver.save_screenshot(str(filepath))
        self.screenshots.append(str(filepath))
        
        return str(filepath)
    
    def run(self) -> TestResult:
        """Run the end-to-end test."""
        start_time = datetime.datetime.now()
        
        # Initialize test result
        result = TestResult(
            test_id=self.test_case.id,
            test_name=self.test_case.name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=start_time,  # Will be updated later
            execution_time=0.0
        )
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Run setup if available
                if self.test_case.setup_function:
                    self.test_case.setup_function()
                
                # Run the test with timeout
                with timeout(self.test_case.timeout_seconds):
                    if self.test_case.test_function:
                        self.test_case.test_function(self)
                
                # Test passed
                result.status = TestStatus.PASSED
            except unittest.SkipTest as e:
                # Test skipped
                result.status = TestStatus.SKIPPED
                result.error_message = str(e)
            except TimeoutError:
                # Test timed out
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {self.test_case.timeout_seconds} seconds"
                
                # Take screenshot on timeout
                if self.driver and self.config.capture_screenshots:
                    self.take_screenshot("timeout")
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
                
                # Take screenshot on failure
                if self.driver and self.config.capture_screenshots:
                    self.take_screenshot("failure")
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
                
                # Take screenshot on error
                if self.driver and self.config.capture_screenshots:
                    self.take_screenshot("error")
            finally:
                # Close browser
                if self.driver:
                    try:
                        self.driver.quit()
                    except Exception as e:
                        logger.error(f"Error closing browser: {e}")
                
                # Run teardown if available
                try:
                    if self.test_case.teardown_function:
                        self.test_case.teardown_function()
                except Exception as e:
                    # Log teardown errors but don't change test status if it passed
                    logger.error(f"Error in teardown: {e}")
                    if result.status == TestStatus.PASSED:
                        result.status = TestStatus.ERROR
                        result.error_message = f"Error in teardown: {e}"
                        result.stack_trace = traceback.format_exc()
        
        # Update test result
        end_time = datetime.datetime.now()
        result.end_time = end_time
        result.execution_time = (end_time - start_time).total_seconds()
        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        result.screenshots = self.screenshots
        
        # Update test case
        self.test_case.status = result.status
        self.test_case.execution_time = result.execution_time
        self.test_case.last_run = end_time
        self.test_case.error_message = result.error_message
        self.test_case.stack_trace = result.stack_trace
        
        return result

class PerformanceTest(BaseTest):
    """Performance test implementation."""
    
    def __init__(self, test_case: TestCase, config: TestConfig = None):
        """Initialize the performance test."""
        super().__init__(config)
        self.test_case = test_case
        self.profiler = None
        self.metrics = {}
    
    def start_profiling(self) -> None:
        """Start profiling."""
        if not PERF_TOOLS_AVAILABLE:
            raise ImportError("Performance testing tools not available")
        
        self.profiler = Profiler()
        self.profiler.start()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and get results."""
        if not self.profiler:
            raise ValueError("Profiler not started")
        
        self.profiler.stop()
        
        # Get profiling results
        results = {
            "text_output": self.profiler.output_text(unicode=True),
            "html_output": self.profiler.output_html()
        }
        
        # Save results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = REPORTS_DIR / f"profile_{self.test_case.id}_{timestamp}.html"
        
        with open(html_path, 'w') as f:
            f.write(results["html_output"])
        
        results["html_file"] = str(html_path)
        
        return results
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure the execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        return result, execution_time
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric."""
        self.metrics[name] = value
    
    def run(self) -> TestResult:
        """Run the performance test."""
        start_time = datetime.datetime.now()
        
        # Initialize test result
        result = TestResult(
            test_id=self.test_case.id,
            test_name=self.test_case.name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=start_time,  # Will be updated later
            execution_time=0.0
        )
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Run setup if available
                if self.test_case.setup_function:
                    self.test_case.setup_function()
                
                # Run the test with timeout
                with timeout(self.test_case.timeout_seconds):
                    if self.test_case.test_function:
                        self.test_case.test_function(self)
                
                # Check performance thresholds
                for metric_name, threshold in self.test_case.expected_results.items():
                    if metric_name in self.metrics:
                        actual_value = self.metrics[metric_name]
                        if actual_value > threshold:
                            raise AssertionError(f"Performance metric '{metric_name}' exceeded threshold: {actual_value} > {threshold}")
                
                # Test passed
                result.status = TestStatus.PASSED
            except unittest.SkipTest as e:
                # Test skipped
                result.status = TestStatus.SKIPPED
                result.error_message = str(e)
            except TimeoutError:
                # Test timed out
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {self.test_case.timeout_seconds} seconds"
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            finally:
                # Run teardown if available
                try:
                    if self.test_case.teardown_function:
                        self.test_case.teardown_function()
                except Exception as e:
                    # Log teardown errors but don't change test status if it passed
                    logger.error(f"Error in teardown: {e}")
                    if result.status == TestStatus.PASSED:
                        result.status = TestStatus.ERROR
                        result.error_message = f"Error in teardown: {e}"
                        result.stack_trace = traceback.format_exc()
        
        # Update test result
        end_time = datetime.datetime.now()
        result.end_time = end_time
        result.execution_time = (end_time - start_time).total_seconds()
        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        result.metrics = self.metrics
        
        # Update test case
        self.test_case.status = result.status
        self.test_case.execution_time = result.execution_time
        self.test_case.last_run = end_time
        self.test_case.error_message = result.error_message
        self.test_case.stack_trace = result.stack_trace
        
        return result

class SecurityTest(BaseTest):
    """Security test implementation."""
    
    def __init__(self, test_case: TestCase, config: TestConfig = None):
        """Initialize the security test."""
        super().__init__(config)
        self.test_case = test_case
        self.vulnerabilities = []
    
    def scan_code(self, path: str) -> Dict[str, Any]:
        """Scan code for security vulnerabilities."""
        if not SECURITY_TOOLS_AVAILABLE:
            raise ImportError("Security testing tools not available")
        
        # Run Bandit for Python code security scanning
        results = {"issues": [], "metrics": {}}
        
        try:
            # Run bandit as a subprocess
            cmd = ["bandit", "-r", path, "-f", "json"]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                # No issues found
                results["metrics"]["total_issues"] = 0
            else:
                # Parse JSON output
                output = json.loads(process.stdout)
                results["issues"] = output.get("results", [])
                results["metrics"] = output.get("metrics", {})
                
                # Add issues to vulnerabilities list
                for issue in results["issues"]:
                    self.vulnerabilities.append({
                        "tool": "bandit",
                        "severity": issue.get("issue_severity", "MEDIUM"),
                        "confidence": issue.get("issue_confidence", "MEDIUM"),
                        "description": issue.get("issue_text", ""),
                        "file": issue.get("filename", ""),
                        "line": issue.get("line_number", 0)
                    })
        except Exception as e:
            logger.error(f"Error running Bandit: {e}")
            results["error"] = str(e)
        
        return results
    
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for security vulnerabilities."""
        if not SECURITY_TOOLS_AVAILABLE:
            raise ImportError("Security testing tools not available")
        
        results = {"vulnerabilities": [], "metrics": {}}
        
        try:
            # Run safety check as a subprocess
            cmd = ["safety", "check", "--json"]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse JSON output
            if process.stdout:
                output = json.loads(process.stdout)
                results["vulnerabilities"] = output.get("vulnerabilities", [])
                results["metrics"]["total_vulnerabilities"] = len(results["vulnerabilities"])
                
                # Add vulnerabilities to list
                for vuln in results["vulnerabilities"]:
                    self.vulnerabilities.append({
                        "tool": "safety",
                        "severity": vuln.get("severity", "MEDIUM"),
                        "package": vuln.get("package_name", ""),
                        "installed_version": vuln.get("installed_version", ""),
                        "vulnerable_spec": vuln.get("vulnerable_spec", ""),
                        "description": vuln.get("advisory", "")
                    })
        except Exception as e:
            logger.error(f"Error running Safety: {e}")
            results["error"] = str(e)
        
        return results
    
    def scan_web_application(self, target_url: str) -> Dict[str, Any]:
        """Scan a web application for security vulnerabilities."""
        if not SECURITY_TOOLS_AVAILABLE:
            raise ImportError("Security testing tools not available")
        
        results = {"alerts": [], "metrics": {}}
        
        try:
            # Initialize ZAP API client
            zap = ZAPv2()
            
            # Start a new session
            zap.core.new_session()
            
            # Access the target URL
            zap.urlopen(target_url)
            
            # Spider the site
            scan_id = zap.spider.scan(target_url)
            
            # Wait for spider to complete
            while int(zap.spider.status(scan_id)) < 100:
                time.sleep(1)
            
            # Run active scan
            scan_id = zap.ascan.scan(target_url)
            
            # Wait for scan to complete
            while int(zap.ascan.status(scan_id)) < 100:
                time.sleep(5)
            
            # Get alerts
            alerts = zap.core.alerts()
            results["alerts"] = alerts
            results["metrics"]["total_alerts"] = len(alerts)
            
            # Add alerts to vulnerabilities list
            for alert in alerts:
                self.vulnerabilities.append({
                    "tool": "zap",
                    "severity": alert.get("risk", "MEDIUM"),
                    "name": alert.get("name", ""),
                    "description": alert.get("description", ""),
                    "url": alert.get("url", ""),
                    "solution": alert.get("solution", "")
                })
        except Exception as e:
            logger.error(f"Error running ZAP scan: {e}")
            results["error"] = str(e)
        
        return results
    
    def run(self) -> TestResult:
        """Run the security test."""
        start_time = datetime.datetime.now()
        
        # Initialize test result
        result = TestResult(
            test_id=self.test_case.id,
            test_name=self.test_case.name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=start_time,  # Will be updated later
            execution_time=0.0
        )
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Run setup if available
                if self.test_case.setup_function:
                    self.test_case.setup_function()
                
                # Run the test with timeout
                with timeout(self.test_case.timeout_seconds):
                    if self.test_case.test_function:
                        self.test_case.test_function(self)
                
                # Check for vulnerabilities
                if self.vulnerabilities:
                    # Count high and critical vulnerabilities
                    high_vulnerabilities = sum(1 for v in self.vulnerabilities 
                                              if v.get("severity", "").upper() in ["HIGH", "CRITICAL"])
                    
                    # Fail if there are high or critical vulnerabilities
                    if high_vulnerabilities > 0:
                        raise AssertionError(f"Found {high_vulnerabilities} high or critical security vulnerabilities")
                
                # Test passed
                result.status = TestStatus.PASSED
            except unittest.SkipTest as e:
                # Test skipped
                result.status = TestStatus.SKIPPED
                result.error_message = str(e)
            except TimeoutError:
                # Test timed out
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {self.test_case.timeout_seconds} seconds"
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            finally:
                # Run teardown if available
                try:
                    if self.test_case.teardown_function:
                        self.test_case.teardown_function()
                except Exception as e:
                    # Log teardown errors but don't change test status if it passed
                    logger.error(f"Error in teardown: {e}")
                    if result.status == TestStatus.PASSED:
                        result.status = TestStatus.ERROR
                        result.error_message = f"Error in teardown: {e}"
                        result.stack_trace = traceback.format_exc()
        
        # Update test result
        end_time = datetime.datetime.now()
        result.end_time = end_time
        result.execution_time = (end_time - start_time).total_seconds()
        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        result.custom_data["vulnerabilities"] = self.vulnerabilities
        result.metrics["total_vulnerabilities"] = len(self.vulnerabilities)
        
        # Update test case
        self.test_case.status = result.status
        self.test_case.execution_time = result.execution_time
        self.test_case.last_run = end_time
        self.test_case.error_message = result.error_message
        self.test_case.stack_trace = result.stack_trace
        
        return result

class LoadTest(BaseTest):
    """Load test implementation."""
    
    def __init__(self, test_case: TestCase, config: TestConfig = None):
        """Initialize the load test."""
        super().__init__(config)
        self.test_case = test_case
        self.metrics = {}
    
    def run_locust_test(self, user_class: Type[HttpUser], host: str, num_users: int, spawn_rate: int, run_time: int) -> Dict[str, Any]:
        """Run a Locust load test."""
        if not PERF_TOOLS_AVAILABLE:
            raise ImportError("Performance testing tools not available")
        
        # Create a temporary stats file
        stats_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        stats_path = stats_file.name
        stats_file.close()
        
        try:
            # Configure Locust
            from locust.env import Environment
            from locust.stats import stats_printer, stats_history
            from locust.log import setup_logging
            
            # Set up environment
            env = Environment(user_classes=[user_class])
            env.create_local_runner()
            
            # Set up logging
            setup_logging("INFO", None)
            
            # Start a greenlet that periodically outputs stats
            stats_printer_greenlet = env.create_web_ui(host, 0)
            
            # Start a greenlet that saves stats history to a file
            stats_history_greenlet = env.runner.stats.start_csv_writer(stats_path)
            
            # Start the test
            env.runner.start(num_users, spawn_rate=spawn_rate)
            
            # Run for the specified time
            time.sleep(run_time)
            
            # Stop the test
            env.runner.quit()
            
            # Process stats
            stats = env.runner.stats
            
            # Extract metrics
            results = {
                "num_requests": stats.total.num_requests,
                "num_failures": stats.total.num_failures,
                "median_response_time": stats.total.median_response_time,
                "avg_response_time": stats.total.avg_response_time,
                "min_response_time": stats.total.min_response_time,
                "max_response_time": stats.total.max_response_time,
                "requests_per_second": stats.total.total_rps,
                "failure_percentage": stats.total.fail_ratio * 100
            }
            
            # Add detailed stats per endpoint
            results["endpoints"] = {}
            for name, stats_entry in stats.entries.items():
                results["endpoints"][name] = {
                    "num_requests": stats_entry.num_requests,
                    "num_failures": stats_entry.num_failures,
                    "median_response_time": stats_entry.median_response_time,
                    "avg_response_time": stats_entry.avg_response_time,
                    "min_response_time": stats_entry.min_response_time,
                    "max_response_time": stats_entry.max_response_time,
                    "requests_per_second": stats_entry.total_rps,
                    "failure_percentage": stats_entry.fail_ratio * 100
                }
            
            # Update metrics
            self.metrics.update(results)
            
            return results
        finally:
            # Clean up
            try:
                os.unlink(stats_path)
            except Exception as e:
                logger.error(f"Error removing stats file: {e}")
    
    def run(self) -> TestResult:
        """Run the load test."""
        start_time = datetime.datetime.now()
        
        # Initialize test result
        result = TestResult(
            test_id=self.test_case.id,
            test_name=self.test_case.name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            end_time=start_time,  # Will be updated later
            execution_time=0.0
        )
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Run setup if available
                if self.test_case.setup_function:
                    self.test_case.setup_function()
                
                # Run the test with timeout
                with timeout(self.test_case.timeout_seconds):
                    if self.test_case.test_function:
                        self.test_case.test_function(self)
                
                # Check performance thresholds
                for metric_name, threshold in self.test_case.expected_results.items():
                    if metric_name in self.metrics:
                        actual_value = self.metrics[metric_name]
                        if actual_value > threshold:
                            raise AssertionError(f"Load test metric '{metric_name}' exceeded threshold: {actual_value} > {threshold}")
                
                # Test passed
                result.status = TestStatus.PASSED
            except unittest.SkipTest as e:
                # Test skipped
                result.status = TestStatus.SKIPPED
                result.error_message = str(e)
            except TimeoutError:
                # Test timed out
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {self.test_case.timeout_seconds} seconds"
            except AssertionError as e:
                # Test failed
                result.status = TestStatus.FAILED
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            except Exception as e:
                # Test error
                result.status = TestStatus.ERROR
                result.error_message = str(e)
                result.stack_trace = traceback.format_exc()
            finally:
                # Run teardown if available
                try:
                    if self.test_case.teardown_function:
                        self.test_case.teardown_function()
                except Exception as e:
                    # Log teardown errors but don't change test status if it passed
                    logger.error(f"Error in teardown: {e}")
                    if result.status == TestStatus.PASSED:
                        result.status = TestStatus.ERROR
                        result.error_message = f"Error in teardown: {e}"
                        result.stack_trace = traceback.format_exc()
        
        # Update test result
        end_time = datetime.datetime.now()
        result.end_time = end_time
        result.execution_time = (end_time - start_time).total_seconds()
        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        result.metrics = self.metrics
        
        # Update test case
        self.test_case.status = result.status
        self.test_case.execution_time = result.execution_time
        self.test_case.last_run = end_time
        self.test_case.error_message = result.error_message
        self.test_case.stack_trace = result.stack_trace
        
        return result

class TestRunner:
    """Runner for executing tests."""
    
    def __init__(self, config: TestConfig = None):
        """Initialize the test runner."""
        self.config = config or TestConfig.load()
        self.test_suites = {}
        self.results = []
    
    def add_test_suite(self, test_suite: TestSuite) -> None:
        """Add a test suite."""
        self.test_suites[test_suite.id] = test_suite
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        return self.test_suites.get(suite_id)
    
    def create_test_case(self, name: str, description: str, test_type: TestType, 
                        test_function: Callable, setup_function: Callable = None, 
                        teardown_function: Callable = None, **kwargs) -> TestCase:
        """Create a new test case."""
        test_id = str(uuid.uuid4())
        
        test_case = TestCase(
            id=test_id,
            name=name,
            description=description,
            test_type=test_type,
            test_function=test_function,
            setup_function=setup_function,
            teardown_function=teardown_function,
            **kwargs
        )
        
        return test_case
    
    def create_test_suite(self, name: str, description: str, test_cases: List[TestCase] = None) -> TestSuite:
        """Create a new test suite."""
        suite_id = str(uuid.uuid4())
        
        test_suite = TestSuite(
            id=suite_id,
            name=name,
            description=description,
            test_cases=test_cases or []
        )
        
        self.add_test_suite(test_suite)
        
        return test_suite
    
    def run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        # Create appropriate test instance based on test type
        if test_case.test_type == TestType.UNIT:
            test = UnitTest(test_case, self.config)
        elif test_case.test_type == TestType.INTEGRATION:
            test = IntegrationTest(test_case, self.config)
        elif test_case.test_type == TestType.END_TO_END:
            test = EndToEndTest(test_case, self.config)
        elif test_case.test_type == TestType.PERFORMANCE:
            test = PerformanceTest(test_case, self.config)
        elif test_case.test_type == TestType.SECURITY:
            test = SecurityTest(test_case, self.config)
        elif test_case.test_type == TestType.LOAD:
            test = LoadTest(test_case, self.config)
        else:
            # Default to unit test for unknown types
            test = UnitTest(test_case, self.config)
        
        # Run the test
        result = test.run()
        
        # Apply self-healing if needed
        if result.status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT] and self.config.enable_self_healing:
            healed_result = self._apply_self_healing(test_case, result)
            if healed_result:
                result = healed_result
        
        # Store the result
        self.results.append(result)
        
        return result
    
    def run_test_suite(self, suite_id: str) -> List[TestResult]:
        """Run all tests in a suite."""
        suite = self.get_test_suite(suite_id)
        if not suite:
            raise ValueError(f"Test suite not found: {suite_id}")
        
        suite_results = []
        
        # Run setup if available
        if suite.setup_function:
            try:
                suite.setup_function()
            except Exception as e:
                logger.error(f"Error in suite setup: {e}")
                # Create a dummy result for the setup failure
                setup_result = TestResult(
                    test_id="setup",
                    test_name=f"{suite.name} - Setup",
                    status=TestStatus.ERROR,
                    start_time=datetime.datetime.now(),
                    end_time=datetime.datetime.now(),
                    execution_time=0.0,
                    error_message=str(e),
                    stack_trace=traceback.format_exc()
                )
                suite_results.append(setup_result)
                return suite_results
        
        # Run tests in parallel or sequentially
        if self.config.parallel_tests and len(suite.test_cases) > 
