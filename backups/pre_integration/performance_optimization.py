#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Sentinel Intelligence AI Platform - Performance Optimization System

This module provides comprehensive performance optimization capabilities for the
Skyscope platform, including real-time monitoring, resource optimization, workload
distribution, caching strategies, database optimization, memory management,
connection pooling, analytics, auto-scaling, reporting, and predictive optimization.

Created on: July 16, 2025
Author: Skyscope Sentinel Intelligence
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import multiprocessing
import asyncio
import psutil
import numpy as np
import pandas as pd
import sqlite3
import requests
import socket
import hashlib
import pickle
import gc
import traceback
import warnings
import datetime
import heapq
import functools
import concurrent.futures
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable, Generator, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, deque, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PerformanceOptimization')

# Constants
OPTIMIZER_VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = "config/performance_optimization.json"
DEFAULT_REPORT_DIR = "performance_reports"
DEFAULT_CACHE_DIR = "cache"
DEFAULT_SAMPLE_INTERVAL = 1.0  # seconds
DEFAULT_AGGREGATION_INTERVAL = 60.0  # seconds
DEFAULT_RETENTION_PERIOD = 86400.0  # seconds (1 day)
DEFAULT_OPTIMIZATION_INTERVAL = 300.0  # seconds (5 minutes)
DEFAULT_PREDICTION_HORIZON = 3600.0  # seconds (1 hour)
DEFAULT_CONNECTION_POOL_SIZE = 10
DEFAULT_THREAD_POOL_SIZE = multiprocessing.cpu_count() * 2
DEFAULT_PROCESS_POOL_SIZE = max(1, multiprocessing.cpu_count() - 1)

# Resource thresholds
DEFAULT_CPU_THRESHOLD = 80.0  # percent
DEFAULT_MEMORY_THRESHOLD = 80.0  # percent
DEFAULT_DISK_THRESHOLD = 90.0  # percent
DEFAULT_NETWORK_THRESHOLD = 80.0  # percent

# Performance metrics
class MetricType(Enum):
    """Enumeration of metric types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    AGENT = "agent"
    CACHE = "cache"
    THREAD = "thread"
    PROCESS = "process"
    CUSTOM = "custom"

@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics"""
    name: str
    type: MetricType
    value: float
    timestamp: float
    unit: str
    source: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceUsage:
    """Data class for storing resource usage information"""
    cpu_percent: float
    memory_percent: float
    memory_used: float
    memory_total: float
    disk_percent: float
    disk_used: float
    disk_total: float
    network_sent: float
    network_recv: float
    timestamp: float
    process_count: int
    thread_count: int
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationAction:
    """Data class for storing optimization actions"""
    action_id: str
    action_type: str
    target: str
    parameters: Dict[str, Any]
    timestamp: float
    status: str
    result: Optional[Dict[str, Any]] = None
    duration: float = 0.0

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    enabled: bool = True
    monitoring_enabled: bool = True
    optimization_enabled: bool = True
    auto_scaling_enabled: bool = True
    prediction_enabled: bool = True
    
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL
    aggregation_interval: float = DEFAULT_AGGREGATION_INTERVAL
    retention_period: float = DEFAULT_RETENTION_PERIOD
    optimization_interval: float = DEFAULT_OPTIMIZATION_INTERVAL
    prediction_horizon: float = DEFAULT_PREDICTION_HORIZON
    
    report_dir: str = DEFAULT_REPORT_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD
    memory_threshold: float = DEFAULT_MEMORY_THRESHOLD
    disk_threshold: float = DEFAULT_DISK_THRESHOLD
    network_threshold: float = DEFAULT_NETWORK_THRESHOLD
    
    connection_pool_size: int = DEFAULT_CONNECTION_POOL_SIZE
    thread_pool_size: int = DEFAULT_THREAD_POOL_SIZE
    process_pool_size: int = DEFAULT_PROCESS_POOL_SIZE
    
    db_optimization_enabled: bool = True
    cache_optimization_enabled: bool = True
    agent_optimization_enabled: bool = True
    memory_optimization_enabled: bool = True
    
    metrics_to_collect: List[str] = field(default_factory=lambda: [
        "cpu.percent", "memory.percent", "disk.percent", "network.bytes_sent", 
        "network.bytes_recv", "process.count", "thread.count", "agent.active",
        "database.queries", "database.query_time", "api.requests", "api.response_time",
        "cache.hits", "cache.misses", "cache.size"
    ])
    
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu.percent": 80.0,
        "memory.percent": 80.0,
        "disk.percent": 90.0,
        "api.response_time": 1.0,  # seconds
        "database.query_time": 0.5  # seconds
    })
    
    custom_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    verbose: bool = False
    debug: bool = False

class PerformanceMonitor:
    """
    Monitors system performance metrics in real-time.
    
    This class collects various performance metrics including CPU, memory,
    disk, network, database, API, agent, cache, thread, and process metrics.
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize the performance monitor.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(config.retention_period / config.sample_interval)))
        self.aggregated_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(config.retention_period / config.aggregation_interval)))
        self.running = False
        self.monitoring_thread = None
        self.aggregation_thread = None
        self.last_sample_time = 0
        self.last_aggregation_time = 0
        self.custom_collectors: Dict[str, Callable[[], Dict[str, float]]] = {}
        
        # Initialize network counters
        self.last_net_io = psutil.net_io_counters()
        self.last_net_io_time = time.time()
        
        # Create directories
        os.makedirs(config.report_dir, exist_ok=True)
        
        logger.info("Performance monitor initialized")
    
    def start(self) -> bool:
        """
        Start the performance monitoring.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Performance monitor already running")
            return False
        
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()
        
        logger.info("Performance monitoring started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the performance monitoring.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Performance monitor not running")
            return False
        
        self.running = False
        
        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
        return True
    
    def register_custom_collector(self, name: str, collector: Callable[[], Dict[str, float]]) -> None:
        """
        Register a custom metric collector.
        
        Args:
            name: Name of the collector
            collector: Function that returns a dictionary of metric name to value
        """
        self.custom_collectors[name] = collector
        logger.info(f"Registered custom collector: {name}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get the current values of all metrics.
        
        Returns:
            Dictionary of metric name to current value
        """
        current_metrics = {}
        
        # Get the most recent value for each metric
        for metric_name, values in self.metrics.items():
            if values:
                current_metrics[metric_name] = values[-1].value
        
        return current_metrics
    
    def get_metric_history(self, metric_name: str, duration: float = None) -> List[Tuple[float, float]]:
        """
        Get the history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            duration: Duration in seconds to look back (None for all available data)
            
        Returns:
            List of (timestamp, value) tuples
        """
        if metric_name not in self.metrics:
            return []
        
        values = list(self.metrics[metric_name])
        
        if duration is not None:
            cutoff_time = time.time() - duration
            values = [m for m in values if m.timestamp >= cutoff_time]
        
        return [(m.timestamp, m.value) for m in values]
    
    def get_aggregated_metrics(self, duration: float = None) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get aggregated metrics for all metrics.
        
        Args:
            duration: Duration in seconds to look back (None for all available data)
            
        Returns:
            Dictionary of metric name to list of (timestamp, value) tuples
        """
        result = {}
        cutoff_time = time.time() - duration if duration is not None else 0
        
        for metric_name, values in self.aggregated_metrics.items():
            result[metric_name] = [(m.timestamp, m.value) for m in values if m.timestamp >= cutoff_time]
        
        return result
    
    def get_resource_usage(self) -> ResourceUsage:
        """
        Get current resource usage.
        
        Returns:
            ResourceUsage object with current resource usage information
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 * 1024 * 1024)  # GB
        memory_total = memory.total / (1024 * 1024 * 1024)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024 * 1024 * 1024)  # GB
        disk_total = disk.total / (1024 * 1024 * 1024)  # GB
        
        # Network usage
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 * 1024)  # MB
        net_recv = net_io.bytes_recv / (1024 * 1024)  # MB
        
        # Process and thread count
        process_count = len(psutil.pids())
        thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_percent=disk_percent,
            disk_used=disk_used,
            disk_total=disk_total,
            network_sent=net_sent,
            network_recv=net_recv,
            timestamp=time.time(),
            process_count=process_count,
            thread_count=thread_count
        )
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects metrics at regular intervals."""
        logger.info("Starting monitoring loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Check if it's time to collect metrics
                if start_time - self.last_sample_time >= self.config.sample_interval:
                    self._collect_metrics()
                    self.last_sample_time = start_time
                
                # Sleep for a short time to avoid high CPU usage
                sleep_time = max(0.01, self.config.sample_interval / 10)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(1)  # Back off on errors
        
        logger.info("Monitoring loop stopped")
    
    def _aggregation_loop(self) -> None:
        """Aggregation loop that computes statistics at regular intervals."""
        logger.info("Starting aggregation loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Check if it's time to aggregate metrics
                if start_time - self.last_aggregation_time >= self.config.aggregation_interval:
                    self._aggregate_metrics()
                    self.last_aggregation_time = start_time
                
                # Sleep for a short time to avoid high CPU usage
                sleep_time = max(0.1, self.config.aggregation_interval / 10)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(5)  # Back off on errors
        
        logger.info("Aggregation loop stopped")
    
    def _collect_metrics(self) -> None:
        """Collect all configured metrics."""
        timestamp = time.time()
        
        # Collect system metrics
        self._collect_system_metrics(timestamp)
        
        # Collect custom metrics
        self._collect_custom_metrics(timestamp)
        
        if self.config.verbose:
            logger.debug(f"Collected metrics at {datetime.datetime.fromtimestamp(timestamp).isoformat()}")
    
    def _collect_system_metrics(self, timestamp: float) -> None:
        """
        Collect system metrics.
        
        Args:
            timestamp: Current timestamp
        """
        # CPU metrics
        if "cpu.percent" in self.config.metrics_to_collect:
            cpu_percent = psutil.cpu_percent(interval=None)
            self._add_metric("cpu.percent", cpu_percent, timestamp, MetricType.CPU, "percent", "system")
        
        if "cpu.count" in self.config.metrics_to_collect:
            cpu_count = psutil.cpu_count(logical=True)
            self._add_metric("cpu.count", cpu_count, timestamp, MetricType.CPU, "cores", "system")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        if "memory.percent" in self.config.metrics_to_collect:
            self._add_metric("memory.percent", memory.percent, timestamp, MetricType.MEMORY, "percent", "system")
        
        if "memory.used" in self.config.metrics_to_collect:
            memory_used_gb = memory.used / (1024 * 1024 * 1024)
            self._add_metric("memory.used", memory_used_gb, timestamp, MetricType.MEMORY, "GB", "system")
        
        if "memory.available" in self.config.metrics_to_collect:
            memory_available_gb = memory.available / (1024 * 1024 * 1024)
            self._add_metric("memory.available", memory_available_gb, timestamp, MetricType.MEMORY, "GB", "system")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        if "disk.percent" in self.config.metrics_to_collect:
            self._add_metric("disk.percent", disk.percent, timestamp, MetricType.DISK, "percent", "system")
        
        if "disk.used" in self.config.metrics_to_collect:
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            self._add_metric("disk.used", disk_used_gb, timestamp, MetricType.DISK, "GB", "system")
        
        if "disk.free" in self.config.metrics_to_collect:
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            self._add_metric("disk.free", disk_free_gb, timestamp, MetricType.DISK, "GB", "system")
        
        # Network metrics
        net_io = psutil.net_io_counters()
        current_time = time.time()
        time_diff = current_time - self.last_net_io_time
        
        if time_diff > 0:
            if "network.bytes_sent" in self.config.metrics_to_collect:
                bytes_sent = (net_io.bytes_sent - self.last_net_io.bytes_sent) / time_diff
                self._add_metric("network.bytes_sent", bytes_sent, timestamp, MetricType.NETWORK, "bytes/s", "system")
            
            if "network.bytes_recv" in self.config.metrics_to_collect:
                bytes_recv = (net_io.bytes_recv - self.last_net_io.bytes_recv) / time_diff
                self._add_metric("network.bytes_recv", bytes_recv, timestamp, MetricType.NETWORK, "bytes/s", "system")
        
        self.last_net_io = net_io
        self.last_net_io_time = current_time
        
        # Process and thread metrics
        if "process.count" in self.config.metrics_to_collect:
            process_count = len(psutil.pids())
            self._add_metric("process.count", process_count, timestamp, MetricType.PROCESS, "count", "system")
        
        if "thread.count" in self.config.metrics_to_collect:
            try:
                thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
                self._add_metric("thread.count", thread_count, timestamp, MetricType.THREAD, "count", "system")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Ignore if processes disappear during iteration
    
    def _collect_custom_metrics(self, timestamp: float) -> None:
        """
        Collect custom metrics from registered collectors.
        
        Args:
            timestamp: Current timestamp
        """
        for collector_name, collector_func in self.custom_collectors.items():
            try:
                metrics = collector_func()
                for metric_name, value in metrics.items():
                    full_name = f"{collector_name}.{metric_name}"
                    self._add_metric(full_name, value, timestamp, MetricType.CUSTOM, "", collector_name)
            except Exception as e:
                logger.error(f"Error collecting custom metrics from {collector_name}: {e}")
                if self.config.debug:
                    logger.debug(traceback.format_exc())
    
    def _add_metric(self, name: str, value: float, timestamp: float, 
                   metric_type: MetricType, unit: str, source: str, 
                   details: Dict[str, Any] = None) -> None:
        """
        Add a metric to the collection.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Metric timestamp
            metric_type: Type of metric
            unit: Unit of measurement
            source: Source of the metric
            details: Additional details (optional)
        """
        metric = PerformanceMetric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=timestamp,
            unit=unit,
            source=source,
            details=details or {}
        )
        
        self.metrics[name].append(metric)
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics for the configured interval."""
        end_time = time.time()
        start_time = end_time - self.config.aggregation_interval
        
        for metric_name, metrics in self.metrics.items():
            # Filter metrics in the aggregation interval
            interval_metrics = [m for m in metrics if start_time <= m.timestamp <= end_time]
            
            if not interval_metrics:
                continue
            
            # Calculate statistics
            values = [m.value for m in interval_metrics]
            count = len(values)
            
            if count > 0:
                avg_value = sum(values) / count
                min_value = min(values)
                max_value = max(values)
                
                # Calculate standard deviation if there's more than one value
                if count > 1:
                    variance = sum((x - avg_value) ** 2 for x in values) / count
                    std_dev = variance ** 0.5
                else:
                    std_dev = 0
                
                # Create aggregated metric
                aggregated_metric = PerformanceMetric(
                    name=metric_name,
                    type=interval_metrics[0].type,
                    value=avg_value,
                    timestamp=end_time,
                    unit=interval_metrics[0].unit,
                    source=interval_metrics[0].source,
                    details={
                        "count": count,
                        "min": min_value,
                        "max": max_value,
                        "std_dev": std_dev,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                )
                
                self.aggregated_metrics[metric_name].append(aggregated_metric)
        
        if self.config.verbose:
            logger.debug(f"Aggregated metrics at {datetime.datetime.fromtimestamp(end_time).isoformat()}")

class ResourceOptimizer:
    """
    Optimizes resource usage based on monitoring data.
    
    This class implements various strategies to optimize CPU, memory,
    disk, and network usage based on real-time performance metrics.
    """
    
    def __init__(self, config: PerformanceConfig, monitor: PerformanceMonitor):
        """
        Initialize the resource optimizer.
        
        Args:
            config: Performance configuration
            monitor: Performance monitor instance
        """
        self.config = config
        self.monitor = monitor
        self.running = False
        self.optimization_thread = None
        self.last_optimization_time = 0
        self.optimization_history: List[OptimizationAction] = []
        self.max_history_size = 1000
        self.optimization_strategies: Dict[str, Callable[[], Optional[OptimizationAction]]] = {
            "cpu": self._optimize_cpu_usage,
            "memory": self._optimize_memory_usage,
            "disk": self._optimize_disk_usage,
            "network": self._optimize_network_usage,
            "database": self._optimize_database_usage,
            "cache": self._optimize_cache_usage,
            "threads": self._optimize_thread_usage,
            "processes": self._optimize_process_usage
        }
        
        logger.info("Resource optimizer initialized")
    
    def start(self) -> bool:
        """
        Start the resource optimization.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Resource optimizer already running")
            return False
        
        if not self.config.optimization_enabled:
            logger.warning("Resource optimization is disabled in configuration")
            return False
        
        self.running = True
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("Resource optimization started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the resource optimization.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Resource optimizer not running")
            return False
        
        self.running = False
        
        # Wait for thread to finish
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        logger.info("Resource optimization stopped")
        return True
    
    def optimize_now(self) -> List[OptimizationAction]:
        """
        Run optimization immediately.
        
        Returns:
            List of optimization actions performed
        """
        logger.info("Running immediate optimization")
        return self._run_optimization_cycle()
    
    def get_optimization_history(self, limit: int = None) -> List[OptimizationAction]:
        """
        Get optimization action history.
        
        Args:
            limit: Maximum number of actions to return (None for all)
            
        Returns:
            List of optimization actions
        """
        if limit is None:
            return list(self.optimization_history)
        else:
            return list(self.optimization_history)[-limit:]
    
    def register_optimization_strategy(self, name: str, strategy: Callable[[], Optional[OptimizationAction]]) -> None:
        """
        Register a custom optimization strategy.
        
        Args:
            name: Strategy name
            strategy: Function that performs optimization and returns an OptimizationAction
        """
        self.optimization_strategies[name] = strategy
        logger.info(f"Registered optimization strategy: {name}")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop that runs at regular intervals."""
        logger.info("Starting optimization loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Check if it's time to run optimization
                if start_time - self.last_optimization_time >= self.config.optimization_interval:
                    self._run_optimization_cycle()
                    self.last_optimization_time = start_time
                
                # Sleep for a short time to avoid high CPU usage
                sleep_time = max(0.1, self.config.optimization_interval / 10)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(5)  # Back off on errors
        
        logger.info("Optimization loop stopped")
    
    def _run_optimization_cycle(self) -> List[OptimizationAction]:
        """
        Run a complete optimization cycle.
        
        Returns:
            List of optimization actions performed
        """
        logger.info("Running optimization cycle")
        actions = []
        
        # Get current resource usage
        resource_usage = self.monitor.get_resource_usage()
        
        # Check CPU usage
        if resource_usage.cpu_percent >= self.config.cpu_threshold:
            action = self._optimize_cpu_usage()
            if action:
                actions.append(action)
        
        # Check memory usage
        if resource_usage.memory_percent >= self.config.memory_threshold:
            action = self._optimize_memory_usage()
            if action:
                actions.append(action)
        
        # Check disk usage
        if resource_usage.disk_percent >= self.config.disk_threshold:
            action = self._optimize_disk_usage()
            if action:
                actions.append(action)
        
        # Run other optimization strategies
        for name, strategy in self.optimization_strategies.items():
            if name not in ["cpu", "memory", "disk"]:  # Skip already handled strategies
                try:
                    action = strategy()
                    if action:
                        actions.append(action)
                except Exception as e:
                    logger.error(f"Error in optimization strategy {name}: {e}")
                    if self.config.debug:
                        logger.debug(traceback.format_exc())
        
        # Add actions to history
        for action in actions:
            self.optimization_history.append(action)
        
        # Trim history if needed
        if len(self.optimization_history) > self.max_history_size:
            self.optimization_history = self.optimization_history[-self.max_history_size:]
        
        logger.info(f"Optimization cycle completed with {len(actions)} actions")
        return actions
    
    def _optimize_cpu_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize CPU usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        logger.info("Optimizing CPU usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        cpu_percent = current_metrics.get("cpu.percent", 0)
        thread_count = current_metrics.get("thread.count", 0)
        
        if cpu_percent >= self.config.cpu_threshold:
            # Create optimization action
            action = OptimizationAction(
                action_id=str(uuid.uuid4()),
                action_type="cpu_optimization",
                target="system",
                parameters={
                    "cpu_percent": cpu_percent,
                    "thread_count": thread_count,
                    "threshold": self.config.cpu_threshold
                },
                timestamp=time.time(),
                status="started"
            )
            
            start_time = time.time()
            
            try:
                # Implement CPU optimization logic
                # For example, reduce thread pool size, adjust process priorities, etc.
                
                # For this example, we'll just simulate an optimization action
                time.sleep(0.1)  # Simulate work
                
                # Update action with result
                action.status = "completed"
                action.result = {
                    "reduced_threads": 0,  # In a real implementation, this would be the actual number
                    "adjusted_priorities": 0,
                    "success": True
                }
            except Exception as e:
                action.status = "failed"
                action.result = {"error": str(e)}
                logger.error(f"CPU optimization failed: {e}")
            
            action.duration = time.time() - start_time
            return action
        
        return None
    
    def _optimize_memory_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize memory usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        logger.info("Optimizing memory usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        memory_percent = current_metrics.get("memory.percent", 0)
        
        if memory_percent >= self.config.memory_threshold:
            # Create optimization action
            action = OptimizationAction(
                action_id=str(uuid.uuid4()),
                action_type="memory_optimization",
                target="system",
                parameters={
                    "memory_percent": memory_percent,
                    "threshold": self.config.memory_threshold
                },
                timestamp=time.time(),
                status="started"
            )
            
            start_time = time.time()
            
            try:
                # Implement memory optimization logic
                # For example, clear caches, run garbage collection, etc.
                
                # Run garbage collection
                collected = gc.collect()
                
                # Update action with result
                action.status = "completed"
                action.result = {
                    "gc_collected": collected,
                    "success": True
                }
            except Exception as e:
                action.status = "failed"
                action.result = {"error": str(e)}
                logger.error(f"Memory optimization failed: {e}")
            
            action.duration = time.time() - start_time
            return action
        
        return None
    
    def _optimize_disk_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize disk usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        logger.info("Optimizing disk usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        disk_percent = current_metrics.get("disk.percent", 0)
        
        if disk_percent >= self.config.disk_threshold:
            # Create optimization action
            action = OptimizationAction(
                action_id=str(uuid.uuid4()),
                action_type="disk_optimization",
                target="system",
                parameters={
                    "disk_percent": disk_percent,
                    "threshold": self.config.disk_threshold
                },
                timestamp=time.time(),
                status="started"
            )
            
            start_time = time.time()
            
            try:
                # Implement disk optimization logic
                # For example, clear temporary files, compress logs, etc.
                
                # For this example, we'll just simulate an optimization action
                time.sleep(0.1)  # Simulate work
                
                # Update action with result
                action.status = "completed"
                action.result = {
                    "space_freed": 0,  # In a real implementation, this would be the actual space freed
                    "success": True
                }
            except Exception as e:
                action.status = "failed"
                action.result = {"error": str(e)}
                logger.error(f"Disk optimization failed: {e}")
            
            action.duration = time.time() - start_time
            return action
        
        return None
    
    def _optimize_network_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize network usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        logger.info("Optimizing network usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        bytes_sent = current_metrics.get("network.bytes_sent", 0)
        bytes_recv = current_metrics.get("network.bytes_recv", 0)
        
        # Check if network usage is high
        network_threshold = self.config.network_threshold * 1024 * 1024  # Convert to bytes/s
        if bytes_sent > network_threshold or bytes_recv > network_threshold:
            # Create optimization action
            action = OptimizationAction(
                action_id=str(uuid.uuid4()),
                action_type="network_optimization",
                target="system",
                parameters={
                    "bytes_sent": bytes_sent,
                    "bytes_recv": bytes_recv,
                    "threshold": network_threshold
                },
                timestamp=time.time(),
                status="started"
            )
            
            start_time = time.time()
            
            try:
                # Implement network optimization logic
                # For example, throttle non-essential connections, batch requests, etc.
                
                # For this example, we'll just simulate an optimization action
                time.sleep(0.1)  # Simulate work
                
                # Update action with result
                action.status = "completed"
                action.result = {
                    "connections_throttled": 0,  # In a real implementation, this would be the actual number
                    "success": True
                }
            except Exception as e:
                action.status = "failed"
                action.result = {"error": str(e)}
                logger.error(f"Network optimization failed: {e}")
            
            action.duration = time.time() - start_time
            return action
        
        return None
    
    def _optimize_database_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize database usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        if not self.config.db_optimization_enabled:
            return None
        
        logger.info("Optimizing database usage")
        
        # Create optimization action
        action = OptimizationAction(
            action_id=str(uuid.uuid4()),
            action_type="database_optimization",
            target="database",
            parameters={},
            timestamp=time.time(),
            status="started"
        )
        
        start_time = time.time()
        
        try:
            # Implement database optimization logic
            # For example, analyze slow queries, optimize indexes, etc.
            
            # For this example, we'll just simulate an optimization action
            time.sleep(0.1)  # Simulate work
            
            # Update action with result
            action.status = "completed"
            action.result = {
                "queries_optimized": 0,  # In a real implementation, this would be the actual number
                "indexes_created": 0,
                "success": True
            }
        except Exception as e:
            action.status = "failed"
            action.result = {"error": str(e)}
            logger.error(f"Database optimization failed: {e}")
        
        action.duration = time.time() - start_time
        return action
    
    def _optimize_cache_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize cache usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        if not self.config.cache_optimization_enabled:
            return None
        
        logger.info("Optimizing cache usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        cache_hits = current_metrics.get("cache.hits", 0)
        cache_misses = current_metrics.get("cache.misses", 0)
        cache_size = current_metrics.get("cache.size", 0)
        
        # Create optimization action
        action = OptimizationAction(
            action_id=str(uuid.uuid4()),
            action_type="cache_optimization",
            target="cache",
            parameters={
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_size": cache_size
            },
            timestamp=time.time(),
            status="started"
        )
        
        start_time = time.time()
        
        try:
            # Implement cache optimization logic
            # For example, adjust cache size, evict least used items, etc.
            
            # For this example, we'll just simulate an optimization action
            time.sleep(0.1)  # Simulate work
            
            # Update action with result
            action.status = "completed"
            action.result = {
                "items_evicted": 0,  # In a real implementation, this would be the actual number
                "cache_resized": False,
                "success": True
            }
        except Exception as e:
            action.status = "failed"
            action.result = {"error": str(e)}
            logger.error(f"Cache optimization failed: {e}")
        
        action.duration = time.time() - start_time
        return action
    
    def _optimize_thread_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize thread usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        logger.info("Optimizing thread usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        thread_count = current_metrics.get("thread.count", 0)
        
        # Create optimization action
        action = OptimizationAction(
            action_id=str(uuid.uuid4()),
            action_type="thread_optimization",
            target="threads",
            parameters={
                "thread_count": thread_count
            },
            timestamp=time.time(),
            status="started"
        )
        
        start_time = time.time()
        
        try:
            # Implement thread optimization logic
            # For example, adjust thread pool sizes, consolidate tasks, etc.
            
            # For this example, we'll just simulate an optimization action
            time.sleep(0.1)  # Simulate work
            
            # Update action with result
            action.status = "completed"
            action.result = {
                "pools_adjusted": 0,  # In a real implementation, this would be the actual number
                "success": True
            }
        except Exception as e:
            action.status = "failed"
            action.result = {"error": str(e)}
            logger.error(f"Thread optimization failed: {e}")
        
        action.duration = time.time() - start_time
        return action
    
    def _optimize_process_usage(self) -> Optional[OptimizationAction]:
        """
        Optimize process usage.
        
        Returns:
            OptimizationAction if optimization was performed, None otherwise
        """
        logger.info("Optimizing process usage")
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        process_count = current_metrics.get("process.count", 0)
        
        # Create optimization action
        action = OptimizationAction(
            action_id=str(uuid.uuid4()),
            action_type="process_optimization",
            target="processes",
            parameters={
                "process_count": process_count
            },
            timestamp=time.time(),
            status="started"
        )
        
        start_time = time.time()
        
        try:
            # Implement process optimization logic
            # For example, adjust process priorities, consolidate processes, etc.
            
            # For this example, we'll just simulate an optimization action
            time.sleep(0.1)  # Simulate work
            
            # Update action with result
            action.status = "completed"
            action.result = {
                "priorities_adjusted": 0,  # In a real implementation, this would be the actual number
                "success": True
            }
        except Exception as e:
            action.status = "failed"
            action.result = {"error": str(e)}
            logger.error(f"Process optimization failed: {e}")
        
        action.duration = time.time() - start_time
        return action

class WorkloadDistributor:
    """
    Manages agent workload distribution.
    
    This class distributes tasks among agents based on their current load,
    capabilities, and system resource availability.
    """
    
    def __init__(self, config: PerformanceConfig, monitor: PerformanceMonitor):
        """
        Initialize the workload distributor.
        
        Args:
            config: Performance configuration
            monitor: Performance monitor instance
        """
        self.config = config
        self.monitor = monitor
        self.agent_loads: Dict[str, float] = {}  # Agent ID to load (0.0-1.0)
        self.agent_capabilities: Dict[str, List[str]] = {}  # Agent ID to list of capabilities
        self.task_queue: List[Dict[str, Any]] = []
        self.running_tasks: Dict[str, Dict[str, Any]] = {}  # Task ID to task info
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}  # Task ID to task info
        self.task_lock = threading.Lock()
        self.distribution_strategies: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {
            "round_robin": self._round_robin_strategy,
            "least_loaded": self._least_loaded_strategy,
            "capability_match": self._capability_match_strategy,
            "priority_based": self._priority_based_strategy
        }
        self.default_strategy = "least_loaded"
        self.last_agent_index = 0
        
        logger.info("Workload distributor initialized")
    
    def register_agent(self, agent_id: str, capabilities: List[str], initial_load: float = 0.0) -> bool:
        """
        Register an agent with the distributor.
        
        Args:
            agent_id: Unique agent ID
            capabilities: List of agent capabilities
            initial_load: Initial load factor (0.0-1.0)
            
        Returns:
            True if registration was successful, False otherwise
        """
        if agent_id in self.agent_loads:
            logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self.agent_loads[agent_id] = max(0.0, min(1.0, initial_load))
        self.agent_capabilities[agent_id] = capabilities
        
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the distributor.
        
        Args:
            agent_id: Agent ID to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        if agent_id not in self.agent_loads:
            logger.warning(f"Agent {agent_id} not registered")
            return False
        
        del self.agent_loads[agent_id]
        del self.agent_capabilities[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def update_agent_load(self, agent_id: str, load: float) -> bool:
        """
        Update an agent's load factor.
        
        Args:
            agent_id: Agent ID
            load: New load factor (0.0-1.0)
            
        Returns:
            True if update was successful, False otherwise
        """
        if agent_id not in self.agent_loads:
            logger.warning(f"Agent {agent_id} not registered")
            return False
        
        self.agent_loads[agent_id] = max(0.0, min(1.0, load))
        return True
    
    def submit_task(self, task: Dict[str, Any]) -> str:
        """
        Submit a task for distribution.
        
        Args:
            task: Task information dictionary
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if "task_id" not in task:
            task["task_id"] = str(uuid.uuid4())
        
        # Add submission time if not provided
        if "submit_time" not in task:
            task["submit_time"] = time.time()
        
        # Add default priority if not provided
        if "priority" not in task:
            task["priority"] = 0
        
        # Add default status
        task["status"] = "queued"
        
        with self.task_lock:
            self.task_queue.append(task)
            
            # Sort queue by priority (higher numbers = higher priority)
            self.task_queue.sort(key=lambda t: t.get("priority", 0), reverse=True)
        
        logger.info(f"Submitted task {task['task_id']} with priority {task['priority']}")
        return task["task_id"]
    
    def distribute_tasks(self, max_tasks: int = None) -> int:
        """
        Distribute queued tasks to agents.
        
        Args:
            max_tasks: Maximum number of tasks to distribute (None for all)
            
        Returns:
            Number of tasks distributed
        """
        distributed_count = 0
        
        with self.task_lock:
            # Determine how many tasks to distribute
            tasks_to_distribute = len(self.task_queue)
            if max_tasks is not None:
                tasks_to_distribute = min(tasks_to_distribute, max_tasks)
            
            # Distribute tasks
            for _ in range(tasks_to_distribute):
                if not self.task_queue:
                    break
                
                task = self.task_queue[0]
                
                # Determine distribution strategy
                strategy = task.get("distribution_strategy", self.default_strategy)
                if strategy not in self.distribution_strategies:
                    strategy = self.default_strategy
                
                # Apply strategy to select agent
                agent_id = self.distribution_strategies[strategy](task)
                
                if agent_id:
                    # Assign task to agent
                    task["assigned_to"] = agent_id
                    task["assign_time"] = time.time()
                    task["status"] = "assigned"
                    
                    # Move task from queue to running
                    self.task_queue.pop(0)
                    self.running_tasks[task["task_id"]] = task
                    
                    # Update agent load
                    load_increase = task.get("estimated_load", 0.1)
                    self.agent_loads[agent_id] = min(1.0, self.agent_loads[agent_id] + load_increase)
                    
                    distributed_count += 1
                    logger.info(f"Assigned task {task['task_id']} to agent {agent_id}")
                else:
                    # No suitable agent found, leave task in queue
                    logger.warning(f"No suitable agent found for task {task['task_id']}")
                    break
        
        logger.info(f"Distributed {distributed_count} tasks")
        return distributed_count
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task ID
            result: Task result information
            
        Returns:
            True if task was marked as completed, False otherwise
        """
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False
            
            task = self.running_tasks[task_id]
            agent_id = task.get("assigned_to")
            
            # Update task information
            task["status"] = "completed"
            task["complete_time"] = time.time()
            task["result"] = result
            
            # Calculate task duration
            if "assign_time" in task:
                task["duration"] = task["complete_time"] - task["assign_time"]
            
            # Move task from running to completed
            del self.running_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            # Update agent load
            if agent_id and agent_id in self.agent_loads:
                load_decrease = task.get("estimated_load", 0.1)
                self.agent_loads[agent_id] = max(0.0, self.agent_loads[agent_id] - load_decrease)
        
        logger.info(f"Completed task {task_id}")
        return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: Task ID
            error: Error information
            
        Returns:
            True if task was marked as failed, False otherwise
        """
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False
            
            task = self.running_tasks[task_id]
            agent_id = task.get("assigned_to")
            
            # Update task information
            task["status"] = "failed"
            task["complete_time"] = time.time()
            task["error"] = error
            
            # Calculate task duration
            if "assign_time" in task:
                task["duration"] = task["complete_time"] - task["assign_time"]
            
            # Move task from running to completed
            del self.running_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            # Update agent load
            if agent_id and agent_id in self.agent_loads:
                load_decrease = task.get("estimated_load", 0.1)
                self.agent_loads[agent_id] = max(0.0, self.agent_loads[agent_id] - load_decrease)
        
        logger.warning(f"Failed task {task_id}: {error}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information dictionary or None if not found
        """
        with self.task_lock:
            # Check queued tasks
            for task in self.task_queue:
                if task["task_id"] == task_id:
                    return task.copy()
            
            # Check running tasks
            if task_id in self.running_tasks:
                return self.running_tasks[task_id].copy()
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].copy()
        
        return None
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status dictionary or None if not found
        """
        if agent_id not in self.agent_loads:
            return None
        
        # Count tasks assigned to this agent
        assigned_tasks = []
        with self.task_lock:
            for task_id, task in self.running_tasks.items():
                if task.get("assigned_to") == agent_id:
                    assigned_tasks.append(task_id)
        
        return {
            "agent_id": agent_id,
            "load": self.agent_loads[agent_id],
            "capabilities": self.agent_capabilities.get(agent_id, []),
            "assigned_tasks": assigned_tasks,
            "task_count": len(assigned_tasks)
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the status of the task queue.
        
        Returns:
            Queue status information
        """
        with self.task_lock:
            return {
                "queued_tasks": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "registered_agents": len(self.agent_loads),
                "available_agents": sum(1 for load in self.agent_loads.values() if load < 0.8)
            }
    
    def register_distribution_strategy(self, name: str, strategy: Callable[[Dict[str, Any]], Optional[str]]) -> None:
        """
        Register a custom distribution strategy.
        
        Args:
            name: Strategy name
            strategy: Function that selects an agent for a task
        """
        self.distribution_strategies[name] = strategy
        logger.info(f"Registered distribution strategy: {name}")
    
    def _round_robin_strategy(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Round-robin distribution strategy.
        
        Args:
            task: Task information
            
        Returns:
            Selected agent ID or None if no agent available
        """
        if not self.agent_loads:
            return None
        
        agent_ids = list(self.agent_loads.keys())
        
        # Start from the last used index
        for i in range(len(agent_ids)):
            index = (self.last_agent_index + i + 1) % len(agent_ids)
            agent_id = agent_ids[index]
            
            # Check if agent is not overloaded
            if self.agent_loads[agent_id] < 0.8:
                self.last_agent_index = index
                return agent_id
        
        # If all agents are overloaded, return None
        return None
    
    def _least_loaded_strategy(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Least-loaded distribution strategy.
        
        Args:
            task: Task information
            
        Returns:
            Selected agent ID or None if no agent available
        """
        if not self.agent_loads:
            return None
        
        # Find agent with lowest load
        min_load = float('inf')
        min_agent = None
        
        for agent_id, load in self.agent_loads.items():
            if load < min_load:
                min_load = load
                min_agent = agent_id
        
        # Check if the least loaded agent is still not too loaded
        if min_agent and min_load < 0.8:
            return min_agent
        
        return None
    
    def _capability_match_strategy(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Capability-matching distribution strategy.
        
        Args:
            task: Task information
            
        Returns:
            Selected agent ID or None if no agent available
        """
        if not self.agent_loads:
            return None
        
        # Get required capabilities from task
        required_capabilities = task.get("required_capabilities", [])
        
        # If no specific capabilities required, use least loaded strategy
        if not required_capabilities:
            return self._least_loaded_strategy(task)
        
        # Find agents with matching capabilities
        matching_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            # Check if agent has all required capabilities
            if all(cap in capabilities for cap in required_capabilities):
                matching_agents.append(agent_id)
        
        if not matching_agents:
            return None
        
        # Among matching agents, find the least loaded one
        min_load = float('inf')
        min_agent = None
        
        for agent_id in matching_agents:
            load = self.agent_loads[agent_id]
            if load < min_load:
                min_load = load
                min_agent = agent_id
        
        # Check if the least loaded agent is still not too loaded
        if min_agent and min_load < 0.8:
            return min_agent
        
        return None
    
    def _priority_based_strategy(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Priority-based distribution strategy.
        
        Args:
            task: Task information
            
        Returns:
            Selected agent ID or None if no agent available
        """
        if not self.agent_loads:
            return None
        
        # Get task priority
        priority = task.get("priority", 0)
        
        # For high-priority tasks, use capability matching if possible
        if priority >= 8:
            agent_id = self._capability_match_strategy(task)
            if agent_id:
                return agent_id
        
        # For medium-priority tasks, use least loaded strategy
        if priority >= 4:
            return self._least_loaded_strategy(task)
        
        # For low-priority tasks, use round robin
        return self._round_robin_strategy(task)

class CacheManager:
    """
    Manages caching strategies for improved performance.
    
    This class implements various caching mechanisms including memory caching,
    disk caching, and distributed caching to reduce computation and I/O overhead.
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize the cache manager.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_info: Dict[str, Dict[str, Any]] = {}
        self.disk_cache_dir = os.path.join(config.cache_dir, "disk_cache")
        self.max_memory_cache_size = 1000
        self.max_disk_cache_size_bytes = 100 * 1024 * 1024  # 100 MB
        self.cache_stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "memory_evictions": 0,
            "disk_evictions": 0
        }
        self.cache_lock = threading.Lock()
        
        # Create cache directories
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        logger.info("Cache manager initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        # Try memory cache first
        value = self._get_from_memory(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self._get_from_disk(key)
        if value is not None:
            # Promote to memory cache
            self._set_in_memory(key, value)
            return value
        
        return default
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        # Set in memory cache
        success = self._set_in_memory(key, value, ttl)
        
        # Also set in disk cache for persistence
        disk_success = self._set_in_disk(key, value, ttl)
        
        return success and disk_success
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        memory_success = self._delete_from_memory(key)
        disk_success = self._delete_from_disk(key)
        
        return memory_success or disk_success
    
    def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if successful, False otherwise
        """
        with self.cache_lock:
            self.memory_cache.clear()
            self.memory_cache_info.clear()
        
        try:
            for filename in os.listdir(self.disk_cache_dir):
                file_path = os.path.join(self.disk_cache_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.cache_lock:
            stats = self.cache_stats.copy()
            stats["memory_cache_size"] = len(self.memory_cache)
            
            # Calculate disk cache size
            disk_cache_size = 0
            try:
                for filename in os.listdir(self.disk_cache_dir):
                    file_path = os.path.join(self.disk_cache_dir, filename)
                    if os.path.isfile(file_path):
                        disk_cache_size += os.path.getsize(file_path)
            except Exception:
                pass
            
            stats["disk_cache_size_bytes"] = disk_cache_size
        
        return stats
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize the cache.
        
        Returns:
            Optimization results
        """
        results = {
            "memory_evicted": 0,
            "disk_evicted": 0,
            "expired_removed": 0
        }
        
        # Remove expired items
        expired = self._remove_expired_items()
        results["expired_removed"] = expired
        
        # Check if memory cache is too large
        with self.cache_lock:
            if len(self.memory_cache) > self.max_memory_cache_size:
                # Evict least recently used items
                to_evict = len(self.memory_cache) - self.max_memory_cache_size
                evicted = self._evict_memory_items(to_evict)
                results["memory_evicted"] = evicted
        
        # Check if disk cache is too large
        disk_cache_size = 0
        try:
            for filename in os.listdir(self.disk_cache_dir):
                file_path = os.path.join(self.disk_cache_dir, filename)
                if os.path.isfile(file_path):
                    disk_cache_size += os.path.getsize(file_path)
        except Exception:
            pass
        
        if disk_cache_size > self.max_disk_cache_size_bytes:
            # Evict least recently used items
            evicted = self._evict_disk_items(disk_cache_size - self.max_disk_cache_size_bytes)
            results["disk_evicted"] = evicted
        
        return results
    
    def _get_from_memory(self, key: str) -> Any:
        """
        Get a value from memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.cache_lock:
            if key in self.memory_cache:
                # Check if expired
                info = self.memory_cache_info.get(key, {})
                expiration = info.get("expiration")
                
                if expiration is None or expiration > time.time():
                    # Update access information
                    info["last_access"] = time.time()
                    info["access_count"] = info.get("access_count", 0) + 1
                    self.memory_cache_info[key] = info
                    
                    # Update stats
                    self.cache_stats["memory_hits"] += 1
                    
                    return self.memory_cache[key]
                else:
                    # Expired, remove from cache
                    del self.memory_cache[key]
                    del self.memory_cache_info[key]
            
            # Not found or expired
            self.cache_stats["memory_misses"] += 1
            return None
    
    def _set_in_memory(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in memory cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.cache_lock:
                # Check if cache is full
                if key not in self.memory_cache and len(self.memory_cache) >= self.max_memory_cache_size:
                    # Evict least recently used item
                    self._evict_memory_items(1)
                
                # Set value
                self.memory_cache[key] = value
                
                # Set metadata
                expiration = time.time() + ttl if ttl is not None else None
                self.memory_cache_info[key] = {
                    "created": time.time(),
                    "last_access": time.time(),
                    "expiration": expiration,
                    "access_count": 0,
                    "size": sys.getsizeof(value)
                }
            
            return True
        except Exception as e:
            logger.error(f"Error setting memory cache key {key}: {e}")
            return False
    
    def _delete_from_memory(self, key: str) -> bool:
        """
        Delete a value from memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        with self.cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                
                if key in self.memory_cache_info:
                    del self.memory_cache_info[key]
                
                return True
        
        return False
    
    def _get_from_disk(self, key: str) -> Any:
        """
        Get a value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Create a filename from the key
        filename = self._get_disk_cache_filename(key)
        file_path = os.path.join(self.disk_cache_dir, filename)
        
        if not os.path.exists(file_path):
            with self.cache_lock:
                self.cache_stats["disk_misses"] += 1
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if expired
            expiration = data.get("expiration")
            if expiration is not None and expiration < time.time():
                # Expired, remove from cache
                os.unlink(file_path)
                with self.cache_lock:
                    self.cache_stats["disk_misses"] += 1
                return None
            
            # Update access time
            os.utime(file_path, None)
            
            with self.cache_lock:
                self.cache_stats["disk_hits"] += 1
            
            return data.get("value")
            
        except Exception as e:
            logger.error(f"Error reading disk cache for key {key}: {e}")
            with self.cache_lock:
                self.cache_stats["disk_misses"] += 1
            return None
    
    def _set_in_disk(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        # Create a filename from the key
        filename = self._get_disk_cache_filename(key)
        file_path = os.path.join(self.disk_cache_dir, filename)
        
        try:
            # Prepare data
            expiration = time.time() + ttl if ttl is not None else None
            data = {
                "key": key,
                "value": value,
                "created": time.time(),
                "expiration": expiration
            }
            
            # Write to file
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            logger.error(f"Error writing disk cache for key {key}: {e}")
            return False
    
    def _delete_from_disk(self, key: str) -> bool:
        """
        Delete a value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        # Create a filename from the key
        filename = self._get_disk_cache_filename(key)
        file_path = os.path.join(self.disk_cache_dir, filename)
        
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                return True
            except Exception as e:
                logger.error(f"Error deleting disk cache for key {key}: {e}")
        
        return False
    
    def _get_disk_cache_filename(self, key: str) -> str:
        """
        Get disk cache filename for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Filename for disk cache
        """
        # Create a hash of the key to use as filename
        hash_obj = hashlib.md5(key.encode())
        return hash_obj.hexdigest() + ".cache"
    
    def _remove_expired_items(self) -> int:
        """
        Remove expired items from cache.
        
        Returns:
            Number of expired items removed
        """
        removed_count = 0
        current_time = time.time()
        
        # Check memory cache
        with self.cache_lock:
            expired_keys = []
            for key, info in self.memory_cache_info.items():
                expiration = info.get("expiration")
                if expiration is not None and expiration < current_time:
                    expired_keys.append(key)
            
            # Remove expired keys
            for key in expired_keys:
                del self.memory_cache[key]
                del self.memory_cache_info[key]
            
            removed_count += len(expired_keys)
        
        # Check disk cache
        try:
            for filename in os.listdir(self.disk_cache_dir):
                file_path = os.path.join(self.disk_cache_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        expiration = data.get("expiration")
                        if expiration is not None and expiration < current_time:
                            os.unlink(file_path)
                            removed_count += 1
                    except Exception:
                        # If we can't read the file, consider it corrupted and remove it
                        try:
                            os.unlink(file_path)
                            removed_count += 1
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Error checking disk cache expiration: {e}")
        
        return removed_count
    
    def _evict_memory_items(self, count: int) -> int:
        """
        Evict items from memory cache.
        
        Args:
            count: Number of items to evict
            
        Returns:
            Number of items evicted
        """
        with self.cache_lock:
            if not self.memory_cache:
                return 0
            
            # Sort items by last access time
            items = list(self.memory_cache_info.items())
            items.sort(key=lambda x: x[1].get("last_access", 0))
            
            # Evict oldest items
            evict_count = min(count, len(items))
            for i in range(evict_count):
                key, _ = items[i]
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    del self.memory_cache_info[key]
            
            self.cache_stats["memory_evictions"] += evict_count
            return evict_count
    
    def _evict_disk_items(self, bytes_to_free: int) -> int:
        """
        Evict items from disk cache.
        
        Args:
            bytes_to_free: Number of bytes to free
            
        Returns:
            Number of items evicted
        """
        try:
            # Get list of cache files with their modification times
            files = []
            for filename in os.listdir(self.disk_cache_dir):
                file_path = os.path.join(self.disk_cache_dir, filename)
                if os.path.isfile(file_path):
                    mtime = os.path.getmtime(file_path)
                    size = os.path.getsize(file_path)
                    files.append((file_path, mtime, size))
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Remove files until we've freed enough space
            evicted = 0
            freed_bytes = 0
            
            for file_path, _, size in files:
                try:
                    os.unlink(file_path)
                    evicted += 1
                    freed_bytes += size
                    
                    if freed_bytes >= bytes_to_free:
                        break
                except Exception:
                    pass
            
            with self.cache_lock:
                self.cache_stats["disk_evictions"] += evicted
            
            return evicted
            
        except Exception as e:
            logger.error(f"Error evicting disk cache items: {e}")
            return 0

class DatabaseOptimizer:
    """
    Optimizes database queries and connections.
    
    This class implements database query optimization, connection pooling,
    and other database performance enhancements.
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize the database optimizer.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.connection