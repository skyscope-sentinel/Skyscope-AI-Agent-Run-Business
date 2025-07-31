# OPTIMIZED BY SYSTEM INTEGRATION
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced GUI Dashboard for Skyscope Sentinel Intelligence
========================================================

Real-time dashboard showing live financial metrics, agent performance,
and cryptocurrency earnings with automatic $1000 interval transfers.

Business: Skyscope Sentinel Intelligence
Version: 2.0.0 Production
"""

import os
import sys
import json
import time
import threading
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# GUI imports
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Installing required GUI packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas", "numpy"])
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    STREAMLIT_AVAILABLE = True

# Configure Streamlit page
st.set_page_config(
    page_title="Skyscope Sentinel Intelligence - Live Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
CUSTOM_CSS = """
<style>
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(75, 94, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(75, 94, 255, 0.1), rgba(0, 163, 255, 0.1));
        border: 1px solid rgba(75, 94, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .agent-status-active {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .agent-status-earning {
        color: #FFD700;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .agent-status-error {
        color: #F44336;
        font-weight: bold;
    }
    
    .income-highlight {
        color: #4CAF50;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    }
    
    .transfer-alert {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 5px rgba(76, 175, 80, 0.5); }
        to { box-shadow: 0 0 20px rgba(76, 175, 80, 0.8); }
    }
    
    .sidebar .sidebar-content {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(75, 94, 255, 0.3);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4b5eff, #00a3ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(75, 94, 255, 0.4);
    }
</style>
"""

class EnhancedGUIDashboard:
    """Enhanced GUI Dashboard for real-time monitoring"""
    
    def __init__(self, swarm_manager):
        """Initialize the dashboard"""
        self.swarm_manager = swarm_manager
        self.update_interval = 5  # Update every 5 seconds
        self.last_update = 0
        
        # Initialize session state
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {
                'income_history': [],
                'agent_history': [],
                'transfer_history': [],
                'performance_history': []
            }
    
    def run(self):
        """Run the dashboard"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # Header
        self.render_header()
        
        # Main dashboard
        self.render_main_dashboard()
        
        # Auto-refresh
        if time.time() - self.last_update > self.update_interval:
            st.rerun()
    
    def render_header(self):
        """Render the dashboard header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="color: #4b5eff; font-size: 3rem; margin: 0;">
                    🚀 SKYSCOPE SENTINEL INTELLIGENCE
                </h1>
                <h3 style="color: #00a3ff; margin: 0.5rem 0;">
                    Autonomous AI Agent Swarm - Live Dashboard
                </h3>
                <p style="color: #888; margin: 0;">
                    Real-time cryptocurrency income generation with 200,000 agents
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_main_dashboard(self):
        """Render the main dashboard content"""
        # Get current metrics
        metrics = self.swarm_manager.get_system_metrics()
        wallet_balances = self.swarm_manager.wallet_manager.update_all_balances()
        
        # Update history
        self.update_history(metrics, wallet_balances)
        
        # Top metrics row
        self.render_top_metrics(metrics, wallet_balances)
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_income_chart()
            self.render_agent_performance_chart()
        
        with col2:
            self.render_wallet_balances_chart(wallet_balances)
            self.render_strategy_performance_chart()
        
        # Agent details
        self.render_agent_details()
        
        # Transfer history
        self.render_transfer_history()
        
        # System controls
        self.render_system_controls()
    
    def render_top_metrics(self, metrics, wallet_balances):
        """Render top-level metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_usd = wallet_balances.get('total_usd', 0.0)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #4CAF50; margin: 0;">💰 Total Balance</h3>
                <div class="income-highlight">${total_usd:.2f}</div>
                <p style="color: #888; margin: 0;">Real-time wallet balance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #00A3FF; margin: 0;">🤖 Active Agents</h3>
                <div style="color: #00A3FF; font-size: 2rem; font-weight: bold;">
                    {metrics.active_agents:,} / {metrics.total_agents:,}
                </div>
                <p style="color: #888; margin: 0;">Agents currently earning</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FFD700; margin: 0;">📈 Daily Income</h3>
                <div style="color: #FFD700; font-size: 2rem; font-weight: bold;">
                    ${metrics.daily_income_usd:.2f}
                </div>
                <p style="color: #888; margin: 0;">Projected daily earnings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF6B6B; margin: 0;">✅ Success Rate</h3>
                <div style="color: #FF6B6B; font-size: 2rem; font-weight: bold;">
                    {metrics.success_rate:.1%}
                </div>
                <p style="color: #888; margin: 0;">Overall task success</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            uptime_days = metrics.uptime_hours / 24
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #9C27B0; margin: 0;">⏱️ Uptime</h3>
                <div style="color: #9C27B0; font-size: 2rem; font-weight: bold;">
                    {uptime_days:.1f}d
                </div>
                <p style="color: #888; margin: 0;">System running time</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Transfer alert
        if metrics.total_income_usd >= 1000:
            st.markdown(f"""
            <div class="transfer-alert">
                🚨 <strong>TRANSFER READY!</strong> 
                ${metrics.total_income_usd:.2f} available for transfer to main wallet
            </div>
            """, unsafe_allow_html=True)
    
    def render_income_chart(self):
        """Render real-time income chart"""
        st.subheader("💰 Real-time Income Generation")
        
        if st.session_state.dashboard_data['income_history']:
            df = pd.DataFrame(st.session_state.dashboard_data['income_history'])
            
            fig = go.Figure()
            
            # Cumulative income line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['total_income'],
                mode='lines+markers',
                name='Total Income',
                line=dict(color='#4CAF50', width=3),
                fill='tonexty'
            ))
            
            # Daily income rate
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['daily_rate'],
                mode='lines',
                name='Daily Rate',
                line=dict(color='#FFD700', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Income Over Time",
                xaxis_title="Time",
                yaxis_title="Total Income ($)",
                yaxis2=dict(
                    title="Daily Rate ($)",
                    overlaying='y',
                    side='right'
                ),
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting income data...")
    
    def render_agent_performance_chart(self):
        """Render agent performance chart"""
        st.subheader("🤖 Agent Performance")
        
        if st.session_state.dashboard_data['agent_history']:
            df = pd.DataFrame(st.session_state.dashboard_data['agent_history'])
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Agent Count', 'Success Rate'),
                vertical_spacing=0.1
            )
            
            # Agent count
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['total_agents'],
                    mode='lines+markers',
                    name='Total Agents',
                    line=dict(color='#00A3FF')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['active_agents'],
                    mode='lines+markers',
                    name='Active Agents',
                    line=dict(color='#4CAF50')
                ),
                row=1, col=1
            )
            
            # Success rate
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['success_rate'],
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color='#FF6B6B')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting agent performance data...")
    
    def render_wallet_balances_chart(self, wallet_balances):
        """Render wallet balances chart"""
        st.subheader("💳 Wallet Balances")
        
        if wallet_balances:
            # Create pie chart of wallet distribution
            wallet_data = []
            for strategy, balance_info in wallet_balances.items():
                if strategy != 'total_usd' and isinstance(balance_info, dict):
                    wallet_data.append({
                        'strategy': strategy.replace('_', ' ').title(),
                        'balance_usd': balance_info.get('balance_usd', 0)
                    })
            
            if wallet_data:
                df = pd.DataFrame(wallet_data)
                df = df[df['balance_usd'] > 0]  # Only show wallets with balance
                
                if not df.empty:
                    fig = px.pie(
                        df,
                        values='balance_usd',
                        names='strategy',
                        title='Wallet Distribution by Strategy'
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label'
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No wallet balances to display")
            else:
                st.info("Loading wallet data...")
        else:
            st.info("Connecting to wallets...")
    
    def render_strategy_performance_chart(self):
        """Render strategy performance comparison"""
        st.subheader("📊 Strategy Performance")
        
        # Get top performing agents by strategy
        top_agents = self.swarm_manager.get_top_performing_agents(50)
        
        if top_agents:
            # Group by strategy
            strategy_data = {}
            for agent in top_agents:
                strategy = agent.strategy.value
                if strategy not in strategy_data:
                    strategy_data[strategy] = {
                        'total_income': 0,
                        'agent_count': 0,
                        'avg_performance': 0
                    }
                
                strategy_data[strategy]['total_income'] += agent.total_income
                strategy_data[strategy]['agent_count'] += 1
                strategy_data[strategy]['avg_performance'] += agent.performance_score
            
            # Calculate averages
            for data in strategy_data.values():
                if data['agent_count'] > 0:
                    data['avg_performance'] /= data['agent_count']
            
            # Create DataFrame
            df = pd.DataFrame([
                {
                    'strategy': strategy.replace('_', ' ').title(),
                    'total_income': data['total_income'],
                    'agent_count': data['agent_count'],
                    'avg_performance': data['avg_performance']
                }
                for strategy, data in strategy_data.items()
            ])
            
            # Create bar chart
            fig = px.bar(
                df,
                x='strategy',
                y='total_income',
                color='avg_performance',
                title='Income by Strategy',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading strategy performance data...")
    
    def render_agent_details(self):
        """Render detailed agent information"""
        st.subheader("🔍 Agent Details")
        
        # Get top performing agents
        top_agents = self.swarm_manager.get_top_performing_agents(20)
        
        if top_agents:
            # Create DataFrame
            agent_data = []
            for agent in top_agents:
                agent_data.append({
                    'Agent ID': agent.agent_id,
                    'Strategy': agent.strategy.value.replace('_', ' ').title(),
                    'Status': agent.status.value.title(),
                    'Total Income': f"${agent.total_income:.2f}",
                    'Tasks Completed': agent.tasks_completed,
                    'Success Rate': f"{agent.success_rate:.1%}",
                    'Performance Score': f"{agent.performance_score:.1f}",
                    'Last Activity': agent.last_activity
                })
            
            df = pd.DataFrame(agent_data)
            
            # Display with custom styling
            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )
        else:
            st.info("Loading agent details...")
    
    def render_transfer_history(self):
        """Render transfer history"""
        st.subheader("💸 Transfer History")
        
        if st.session_state.dashboard_data['transfer_history']:
            df = pd.DataFrame(st.session_state.dashboard_data['transfer_history'])
            
            # Display recent transfers
            st.dataframe(
                df.tail(10),
                use_container_width=True,
                height=300
            )
            
            # Total transferred
            total_transferred = df['amount'].sum()
            st.metric("Total Transferred", f"${total_transferred:.2f}")
        else:
            st.info("No transfers yet")
    
    def render_system_controls(self):
        """Render system control panel"""
        st.subheader("⚙️ System Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Scale Up Agents"):
                current_count = len(self.swarm_manager.agents)
                new_target = min(current_count + 100, self.swarm_manager.max_agents)
                self.swarm_manager.scale_agents(new_target)
                st.success(f"Scaling to {new_target} agents")
        
        with col2:
            if st.button("📉 Scale Down Agents"):
                current_count = len(self.swarm_manager.agents)
                new_target = max(current_count - 100, 100)
                self.swarm_manager.scale_agents(new_target)
                st.success(f"Scaling to {new_target} agents")
        
        with col3:
            if st.button("🔄 Optimize Allocation"):
                self.swarm_manager.optimize_agent_allocation()
                st.success("Agent allocation optimized")
        
        with col4:
            if st.button("💰 Force Transfer Check"):
                self.swarm_manager.check_transfer_threshold()
                st.success("Transfer check completed")
        
        # System configuration
        with st.expander("System Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Configuration:**")
                st.json({
                    "Max Agents": self.swarm_manager.max_agents,
                    "Transfer Threshold": "$1,000",
                    "Update Interval": f"{self.update_interval}s",
                    "Business Name": "Skyscope Sentinel Intelligence"
                })
            
            with col2:
                st.write("**Environment Status:**")
                env_status = {
                    "Infura API": "✅" if os.environ.get("INFURA_API_KEY") else "❌",
                    "Wallet Seed": "✅" if os.environ.get("SKYSCOPE_WALLET_SEED_PHRASE") else "❌",
                    "BTC Address": "✅" if os.environ.get("DEFAULT_BTC_ADDRESS") else "❌",
                    "ETH Address": "✅" if os.environ.get("DEFAULT_ETH_ADDRESS") else "❌"
                }
                
                for key, status in env_status.items():
                    st.write(f"{key}: {status}")
    
    def update_history(self, metrics, wallet_balances):
        """Update historical data"""
        current_time = datetime.datetime.now()
        
        # Update income history
        st.session_state.dashboard_data['income_history'].append({
            'timestamp': current_time,
            'total_income': metrics.total_income_usd,
            'daily_rate': metrics.daily_income_usd
        })
        
        # Update agent history
        st.session_state.dashboard_data['agent_history'].append({
            'timestamp': current_time,
            'total_agents': metrics.total_agents,
            'active_agents': metrics.active_agents,
            'success_rate': metrics.success_rate
        })
        
        # Keep only last 100 data points
        for key in ['income_history', 'agent_history']:
            if len(st.session_state.dashboard_data[key]) > 100:
                st.session_state.dashboard_data[key] = st.session_state.dashboard_data[key][-100:]
        
        # Update transfer history if there's a new transfer
        if metrics.last_transfer_amount > 0:
            # Check if this transfer is already recorded
            existing_transfers = st.session_state.dashboard_data['transfer_history']
            if not existing_transfers or existing_transfers[-1]['time'] != metrics.last_transfer_time:
                st.session_state.dashboard_data['transfer_history'].append({
                    'time': metrics.last_transfer_time,
                    'amount': metrics.last_transfer_amount,
                    'type': 'Automatic Transfer'
                })
        
        self.last_update = time.time()

# Standalone dashboard runner
def run_dashboard():
    """Run the dashboard as a standalone application"""
    try:
        # Try to import the swarm manager
        from core_autonomous_system import AgentSwarmManager
        
        # Initialize swarm manager
        swarm_manager = AgentSwarmManager()
        
        # Create some demo agents if none exist
        if len(swarm_manager.agents) == 0:
            swarm_manager.create_agents(10)  # Create 10 demo agents
        
        # Initialize and run dashboard
        dashboard = EnhancedGUIDashboard(swarm_manager)
        dashboard.run()
    
    except ImportError:
        st.error("Core autonomous system not available. Please ensure core_autonomous_system.py is in the same directory.")
    except Exception as e:
        st.error(f"Error running dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()
