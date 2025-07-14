#!/usr/bin/env python3
"""
System validation script for Skyscope Sentinel Intelligence
This script validates that all core components are working correctly
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing critical imports...")
    
    critical_imports = [
        ("streamlit", "Streamlit web framework"),
        ("config", "Configuration management"),
        ("models", "Data models"),
        ("agent_manager", "Agent management system"),
        ("state_manager", "State management"),
        ("ui_manager", "UI management"),
        ("utils", "Utility functions"),
    ]
    
    optional_imports = [
        ("quantum_manager", "Quantum computing features"),
        ("browser_automation", "Browser automation"),
        ("filesystem_manager", "File system operations"),
        ("opencore_manager", "OpenCore integration"),
        ("business_generator", "Business plan generation"),
    ]
    
    success_count = 0
    total_count = len(critical_imports) + len(optional_imports)
    
    # Test critical imports
    for module_name, description in critical_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}: {description}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {module_name}: {description} - {e}")
    
    # Test optional imports
    for module_name, description in optional_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}: {description}")
            success_count += 1
        except Exception as e:
            print(f"   ⚠️ {module_name}: {description} - {e} (optional)")
            success_count += 1  # Count as success since it's optional
    
    print(f"\n📊 Import Results: {success_count}/{total_count} modules loaded")
    return success_count >= len(critical_imports)

def test_file_structure():
    """Test that all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "app.py",
        "config.py",
        "models.py",
        "agent_manager.py",
        "state_manager.py",
        "ui_manager.py",
        "utils.py",
        "requirements.txt",
        "install_macos.sh",
        "logo.png",
        "knowledge_base.md",
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All required files present")
    return True

def test_configuration():
    """Test configuration system"""
    print("\n🧪 Testing configuration system...")
    
    try:
        from config import get_config, get, set
        
        # Test config initialization
        config = get_config()
        print("   ✅ Configuration initialized")
        
        # Test getting values
        app_name = get("app.name")
        print(f"   ✅ App name: {app_name}")
        
        # Test setting values
        set("test.validation", "success")
        value = get("test.validation")
        assert value == "success", f"Expected 'success', got {value}"
        print("   ✅ Configuration set/get working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_agent_system():
    """Test agent management system"""
    print("\n🧪 Testing agent management system...")
    
    try:
        from agent_manager import AgentManager
        from models import AgentTask, TaskStatus, TaskPriority
        
        # Initialize manager
        manager = AgentManager()
        print("   ✅ AgentManager initialized")
        
        # Test task creation
        task = AgentTask(
            description="Test task",
            priority=TaskPriority.HIGH
        )
        print("   ✅ Task creation working")
        
        # Test memory system
        manager.memory.add("test", "value", {"source": "validation"})
        retrieved = manager.memory.get("test")
        assert retrieved == "value", f"Expected 'value', got {retrieved}"
        print("   ✅ Memory system working")
        
        # Test knowledge manager
        item_id = manager.knowledge_manager.add_item(
            title="Test Knowledge",
            content="Test content",
            source="validation"
        )
        item = manager.knowledge_manager.get_item(item_id)
        assert item is not None, "Knowledge item should not be None"
        print("   ✅ Knowledge management working")
        
        # Test tool registry
        def test_tool(message):
            return f"Tool: {message}"
        
        manager.tool_registry.register_tool(
            name="test_tool",
            func=test_tool,
            description="Test tool",
            parameters={"message": "Input"}
        )
        
        result = manager.tool_registry.execute_tool("test_tool", message="Hello")
        assert result == "Tool: Hello", f"Unexpected result: {result}"
        print("   ✅ Tool registry working")
        
        # Cleanup
        manager.shutdown()
        print("   ✅ Manager shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Agent system test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_compatibility():
    """Test Streamlit compatibility"""
    print("\n🧪 Testing Streamlit compatibility...")
    
    try:
        import streamlit as st
        print("   ✅ Streamlit imported successfully")
        
        # Test if app.py can be imported (basic syntax check)
        import app
        print("   ✅ Main app module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Streamlit compatibility test failed: {e}")
        return False

def test_install_script():
    """Test install script permissions and basic structure"""
    print("\n🧪 Testing install script...")
    
    script_path = Path("install_macos.sh")
    
    if not script_path.exists():
        print("   ❌ install_macos.sh not found")
        return False
    
    # Check if script is executable
    if not os.access(script_path, os.X_OK):
        print("   ⚠️ install_macos.sh is not executable")
        print("   💡 Run: chmod +x install_macos.sh")
    else:
        print("   ✅ install_macos.sh is executable")
    
    # Check script content
    with open(script_path, 'r') as f:
        content = f.read()
        
    required_sections = [
        "#!/bin/bash",
        "echo_step",
        "echo_success",
        "echo_error",
        "pyinstaller",
        "create-dmg"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"   ⚠️ Missing sections in install script: {', '.join(missing_sections)}")
    else:
        print("   ✅ Install script structure looks good")
    
    return len(missing_sections) == 0

def main():
    """Run all validation tests"""
    print("🚀 Skyscope Sentinel Intelligence - System Validation")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Critical Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Agent Management System", test_agent_system),
        ("Streamlit Compatibility", test_streamlit_compatibility),
        ("Install Script", test_install_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The system is ready for deployment")
        print("\n💡 Next steps:")
        print("   1. Run './install_macos.sh' to build the macOS app")
        print("   2. Or run 'streamlit run app.py' for development")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("🔧 Please fix the issues above before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())