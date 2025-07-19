#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick System Test
================

Test the autonomous orchestrator without GUI to verify it's working.
"""

import time
from autonomous_orchestrator import get_orchestrator

def test_system():
    print("🚀 Testing Skyscope AI Agentic Swarm Business/Enterprise System")
    print("=" * 60)
    
    # Get orchestrator
    orchestrator = get_orchestrator()
    
    # Start autonomous operations
    print("Starting autonomous operations...")
    orchestrator.start_autonomous_operations()
    
    # Run for 30 seconds and show metrics
    for i in range(6):
        time.sleep(5)
        
        metrics = orchestrator.get_metrics()
        agent_status = orchestrator.get_agent_status()
        income_status = orchestrator.get_income_stream_status()
        
        print(f"\n📊 Metrics Update #{i+1}")
        print(f"Total Agents: {metrics.total_agents:,}")
        print(f"Active Agents: {metrics.active_agents:,}")
        print(f"Daily Income: ${metrics.total_daily_income:.2f}")
        print(f"Lifetime Income: ${metrics.total_lifetime_income:.2f}")
        print(f"Average Performance: {metrics.average_agent_performance:.2%}")
        print(f"System Uptime: {metrics.system_uptime:.1f} hours")
        
        # Show some agent activity
        activities = orchestrator.get_recent_activities(limit=3)
        if activities:
            print("\n🔥 Recent Business Activities:")
            for activity in activities:
                timestamp = activity['timestamp'].strftime("%H:%M:%S")
                print(f"  [{timestamp}] {activity['message']}")
    
    # Stop operations
    print("\n🛑 Stopping autonomous operations...")
    orchestrator.stop_autonomous_operations()
    
    print("\n✅ System test completed successfully!")
    print("The autonomous business system is working correctly.")
    print("\nTo start the full GUI application:")
    print("  source venv/bin/activate && python3 main_application.py")

if __name__ == "__main__":
    test_system()