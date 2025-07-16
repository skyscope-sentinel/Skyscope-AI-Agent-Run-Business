#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Monitoring Configuration
===========================================================

This module provides configuration settings for the Skyscope Sentinel
Intelligence AI monitoring dashboard and metrics collection system.

The configuration includes:
- Dashboard theme and styling
- Layout configuration
- System monitoring settings
- Agent monitoring settings
- Business metrics settings
- Security monitoring settings
- Crypto portfolio settings
- Data visualization options
"""

import os
from pathlib import Path

# --- Base Paths and Directories ---

# Base directory for monitoring data
MONITORING_BASE_DIR = Path("monitoring_data")
MONITORING_BASE_DIR.mkdir(exist_ok=True, parents=True)

# Logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# --- Dashboard Theme and Styling ---

THEME = {
    'dark': True,
    'primary': '#007BFF',  # Blue
    'secondary': '#6C757D',
    'success': '#28A745',  # Green
    'info': '#17A2B8',     # Cyan
    'warning': '#FFC107',  # Yellow
    'danger': '#DC3545',   # Red
    'light': '#F8F9FA',
    'dark': '#343A40',
    'background': '#121212',
    'text': '#FFFFFF',
    'neon': {
        'blue': '#00FFFF',
        'green': '#00FF00',
        'pink': '#FF00FF',
        'yellow': '#FFFF00',
        'orange': '#FF8C00'
    }
}

# --- Dashboard Layout Configuration ---

DASHBOARD_CONFIG = {
    'refresh_interval': 5,  # seconds
    'history_length': 100,  # data points to keep in memory
    'default_timeframe': '1h',  # default time frame for charts
    'available_timeframes': ['5m', '15m', '1h', '4h', '1d', '1w'],
    'max_agents_display': 50,  # maximum number of agents to display in the agent view
    'anomaly_threshold': 2.5,  # standard deviations for anomaly detection
    'enable_3d_charts': True,  # enable 3D visualizations
    'enable_websockets': True,  # enable WebSocket real-time updates
    'websocket_port': 8765,    # WebSocket server port
    'dark_mode': True,         # use dark mode for dashboard
    'chart_animation': True,   # enable chart animations
    'responsive_layout': True, # enable responsive layout for different screen sizes
    'show_tooltips': True,     # show tooltips on charts
    'auto_refresh': True,      # automatically refresh data
    'max_data_points': 10000,  # maximum number of data points to store in memory
}

# --- System Monitoring Settings ---

SYSTEM_MONITORING = {
    'enabled': True,
    'collection_interval': 2,  # seconds
    'metrics': [
        'cpu_usage',
        'memory_usage',
        'disk_usage',
        'network_sent',
        'network_recv',
        'gpu_usage',
        'gpu_memory',
        'process_count',
        'thread_count',
        'temperature'  # if available
    ],
    'log_to_file': True,
    'log_file': LOGS_DIR / 'system_metrics.log',
    'alert_thresholds': {
        'cpu_usage': 90,       # percent
        'memory_usage': 90,    # percent
        'disk_usage': 90,      # percent
        'temperature': 80      # degrees Celsius
    },
    'historical_data': {
        'retention_period': 30,  # days
        'aggregation_interval': 5  # minutes
    }
}

# --- Agent Monitoring Settings ---

AGENT_MONITORING = {
    'enabled': True,
    'collection_interval': 5,  # seconds
    'max_agents': 10000,       # maximum number of agents to track
    'metrics': [
        'status',
        'response_time',
        'success_rate',
        'cpu_usage',
        'memory_usage',
        'task_count',
        'error_count',
        'uptime'
    ],
    'log_to_file': True,
    'log_file': LOGS_DIR / 'agent_metrics.log',
    'alert_thresholds': {
        'error_rate': 0.1,     # 10% errors
        'response_time': 5000  # milliseconds
    },
    'historical_data': {
        'retention_period': 14,  # days
        'aggregation_interval': 15  # minutes
    },
    'agent_types': [
        'business',
        'trading',
        'research',
        'security',
        'analytics',
        'content',
        'customer_service',
        'development',
        'infrastructure',
        'quantum'
    ]
}

# --- Business Metrics Settings ---

BUSINESS_METRICS = {
    'enabled': True,
    'collection_interval': 15,  # minutes
    'metrics': [
        'revenue',
        'expenses',
        'profit',
        'user_count',
        'transaction_count',
        'conversion_rate',
        'customer_acquisition_cost',
        'lifetime_value',
        'churn_rate',
        'active_users'
    ],
    'log_to_file': True,
    'log_file': LOGS_DIR / 'business_metrics.log',
    'alert_thresholds': {
        'profit_margin': 0.1,  # 10% profit margin
        'churn_rate': 0.05     # 5% churn rate
    },
    'historical_data': {
        'retention_period': 365,  # days
        'aggregation_interval': 60  # minutes
    },
    'forecast_periods': 12,  # number of periods to forecast
    'forecast_methods': [
        'moving_average',
        'exponential_smoothing',
        'linear_regression',
        'arima'
    ]
}

# --- Security Monitoring Settings ---

SECURITY_MONITORING = {
    'enabled': True,
    'collection_interval': 1,  # seconds
    'max_events': 10000,       # maximum number of security events to store
    'event_types': [
        'authentication',
        'authorization',
        'network',
        'system',
        'application',
        'data',
        'api',
        'user'
    ],
    'severity_levels': [
        'info',
        'low',
        'medium',
        'high',
        'critical'
    ],
    'log_to_file': True,
    'log_file': LOGS_DIR / 'security_events.log',
    'alert_notification': {
        'email': True,
        'sms': False,
        'webhook': True
    },
    'historical_data': {
        'retention_period': 90,  # days
        'aggregation_interval': 10  # minutes
    },
    'threat_intelligence': {
        'enabled': True,
        'update_interval': 3600  # seconds
    }
}

# --- Crypto Portfolio Settings ---

CRYPTO_PORTFOLIO = {
    'enabled': True,
    'update_interval': 300,  # seconds
    'supported_blockchains': [
        'ethereum',
        'bitcoin',
        'bsc',
        'polygon',
        'avalanche',
        'arbitrum',
        'optimism',
        'solana',
        'cardano'
    ],
    'price_data_sources': [
        'coingecko',
        'coinmarketcap',
        'binance',
        'kraken'
    ],
    'historical_data': {
        'retention_period': 365,  # days
        'price_update_interval': 3600  # seconds
    },
    'defi_protocols': [
        'uniswap',
        'aave',
        'compound',
        'curve',
        'yearn',
        'sushiswap',
        'balancer'
    ],
    'alert_thresholds': {
        'price_change': 10,  # percent
        'portfolio_value_change': 15  # percent
    }
}

# --- Data Visualization Options ---

VISUALIZATION_OPTIONS = {
    'chart_types': {
        'time_series': 'line',
        'distribution': 'bar',
        'composition': 'pie',
        'correlation': 'scatter',
        'hierarchy': 'treemap',
        'network': 'network',
        'geospatial': 'map'
    },
    'color_schemes': {
        'categorical': [
            THEME['neon']['blue'],
            THEME['neon']['green'],
            THEME['neon']['pink'],
            THEME['neon']['yellow'],
            THEME['neon']['orange']
        ],
        'sequential': [
            '#0d0887',
            '#41049d',
            '#6a00a8',
            '#8f0da4',
            '#b12a90',
            '#cc4778',
            '#e16462',
            '#f2844b',
            '#fca636',
            '#fcce25',
            '#f0f921'
        ],
        'diverging': [
            '#0d0887',
            '#46039f',
            '#7201a8',
            '#9c179e',
            '#bd3786',
            '#d8576b',
            '#ed7953',
            '#fb9f3a',
            '#fdca26',
            '#f0f921'
        ]
    },
    'default_layout': {
        'font': {
            'family': 'Arial, sans-serif',
            'size': 12,
            'color': THEME['text']
        },
        'paper_bgcolor': THEME['background'],
        'plot_bgcolor': THEME['background'],
        'autosize': True,
        'margin': {
            'l': 50,
            'r': 50,
            'b': 50,
            't': 50,
            'pad': 4
        },
        'xaxis': {
            'gridcolor': '#444444',
            'zerolinecolor': '#666666'
        },
        'yaxis': {
            'gridcolor': '#444444',
            'zerolinecolor': '#666666'
        }
    },
    '3d_options': {
        'enabled': DASHBOARD_CONFIG['enable_3d_charts'],
        'orbital_rotation': True,
        'auto_rotate': False,
        'projection_type': 'perspective'  # or 'orthographic'
    }
}

# --- Export Options ---

EXPORT_OPTIONS = {
    'formats': [
        'json',
        'csv',
        'excel',
        'pdf',
        'png'
    ],
    'default_format': 'json',
    'include_metadata': True,
    'compression': True,
    'max_export_size': 1000000  # maximum number of records to export
}

# --- Integration Options ---

INTEGRATION_OPTIONS = {
    'quantum_enhanced_ai': {
        'enabled': True,
        'module_path': 'quantum_enhanced_ai'
    },
    'blockchain_crypto': {
        'enabled': True,
        'module_path': 'blockchain_crypto_integration'
    },
    'external_apis': {
        'enabled': True,
        'rate_limit': 100,  # requests per minute
        'timeout': 30  # seconds
    },
    'notification_services': {
        'email': {
            'enabled': True,
            'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.example.com'),
            'smtp_port': int(os.environ.get('SMTP_PORT', 587)),
            'sender_email': os.environ.get('SENDER_EMAIL', 'alerts@skyscope.ai'),
            'use_tls': True
        },
        'webhook': {
            'enabled': True,
            'endpoints': [
                os.environ.get('WEBHOOK_URL', 'https://example.com/webhook')
            ]
        }
    }
}
