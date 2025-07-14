#!/usr/bin/env python3
"""
Test script for AgentManager functionality
"""

import sys
import traceback
from agent_manager import AgentManager, AgentRole
from models import AgentTask, TaskPriority, TaskStatus

def test_agent_manager():
    """Test basic AgentManager functionality"""
    print("🧪 Testing AgentManager...")
    
    try:
        # Initialize AgentManager
        print("1. Initializing AgentManager...")
        manager = AgentManager()
        print("   ✅ AgentManager initialized successfully")
        
        # Test memory functionality
        print("2. Testing memory functionality...")
        manager.memory.add("test_key", "test_value", {"source": "test"})
        retrieved = manager.memory.get("test_key")
        assert retrieved == "test_value", f"Expected 'test_value', got {retrieved}"
        print("   ✅ Memory functionality working")
        
        # Test knowledge manager
        print("3. Testing knowledge manager...")
        item_id = manager.knowledge_manager.add_item(
            title="Test Knowledge",
            content="This is test knowledge content",
            source="test_script"
        )
        item = manager.knowledge_manager.get_item(item_id)
        assert item is not None, "Knowledge item should not be None"
        assert item["title"] == "Test Knowledge", f"Expected 'Test Knowledge', got {item['title']}"
        print("   ✅ Knowledge manager working")
        
        # Test tool registry
        print("4. Testing tool registry...")
        def test_tool(message):
            return f"Tool executed: {message}"
        
        manager.tool_registry.register_tool(
            name="test_tool",
            func=test_tool,
            description="A test tool",
            parameters={"message": "Input message"}
        )
        
        result = manager.tool_registry.execute_tool("test_tool", message="Hello World")
        assert result == "Tool executed: Hello World", f"Unexpected tool result: {result}"
        print("   ✅ Tool registry working")
        
        # Test task creation
        print("5. Testing task creation...")
        task = AgentTask(
            description="Test task",
            input_data="test input",
            priority=TaskPriority.HIGH
        )
        print(f"   Debug: task.status = {task.status}, type = {type(task.status)}")
        print(f"   Debug: TaskStatus.PENDING = {TaskStatus.PENDING}, type = {type(TaskStatus.PENDING)}")
        print(f"   Debug: Equal? {task.status == TaskStatus.PENDING}, Identical? {task.status is TaskStatus.PENDING}")
        
        if task.status == TaskStatus.PENDING and task.priority == TaskPriority.HIGH:
            print("   ✅ Task creation working")
        else:
            raise AssertionError(f"Task creation failed: status={task.status}, priority={task.priority}")
        
        # Test task status updates
        print("6. Testing task status updates...")
        task.update_status(TaskStatus.IN_PROGRESS)
        if task.status != TaskStatus.IN_PROGRESS:
            raise AssertionError(f"Expected TaskStatus.IN_PROGRESS, got {task.status}")
        assert task.started_at is not None, "started_at should be set"
        print("   ✅ Task status updates working")
        
        # Test pipeline creation
        print("7. Testing pipeline creation...")
        from agent_manager import Pipeline
        pipeline = Pipeline(
            name="Test Pipeline",
            description="A test pipeline",
            tasks=[task]
        )
        assert len(pipeline.tasks) == 1, f"Expected 1 task, got {len(pipeline.tasks)}"
        print("   ✅ Pipeline creation working")
        
        # Test agent creation (will fail without swarms but should handle gracefully)
        print("8. Testing agent creation (expected to fail without swarms)...")
        try:
            agent_id = manager.create_agent(
                agent_name="Test Agent",
                role=AgentRole.RESEARCHER
            )
            print(f"   ⚠️ Agent creation unexpectedly succeeded: {agent_id}")
        except ImportError as e:
            print(f"   ✅ Agent creation failed as expected: {e}")
        
        # Test shutdown
        print("9. Testing manager shutdown...")
        manager.shutdown()
        print("   ✅ Manager shutdown completed")
        
        print("\n🎉 All tests passed! AgentManager is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration functionality"""
    print("\n🧪 Testing Configuration...")
    
    try:
        from config import get_config, get, set
        
        # Test config initialization
        print("1. Testing config initialization...")
        config = get_config()
        assert config is not None, "Config should not be None"
        print("   ✅ Config initialized successfully")
        
        # Test getting default values
        print("2. Testing default config values...")
        app_name = get("app.name")
        assert app_name is not None, "App name should not be None"
        print(f"   ✅ App name: {app_name}")
        
        # Test setting and getting values
        print("3. Testing config set/get...")
        set("test.value", "test_data")
        retrieved = get("test.value")
        assert retrieved == "test_data", f"Expected 'test_data', got {retrieved}"
        print("   ✅ Config set/get working")
        
        print("🎉 Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting AgentManager Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test configuration
    if not test_config():
        success = False
    
    # Test agent manager
    if not test_agent_manager():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The system is ready for use.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()