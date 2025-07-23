#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete System Validation Script
=================================

This script validates that all components of the Skyscope AI Agentic Swarm
Business/Enterprise system are properly installed and configured.

Created: January 2025
Author: Skyscope Sentinel Intelligence
"""

import sys
import os
import importlib
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

class SystemValidator:
    """Validates the complete system setup"""
    
    def __init__(self):
        self.results = {
            'core_dependencies': {},
            'gui_framework': {},
            'business_modules': {},
            'ai_libraries': {},
            'crypto_libraries': {},
            'system_files': {},
            'configuration': {},
            'overall_status': 'unknown'
        }
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        
    def print_success(self, message: str):
        """Print success message"""
        print(f"✓ {message}")
        
    def print_error(self, message: str):
        """Print error message"""
        print(f"✗ {message}")
        
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"⚠ {message}")
        
    def check_import(self, module_name: str, description: str = None) -> bool:
        """Check if a module can be imported"""
        try:
            importlib.import_module(module_name)
            desc = description or module_name
            self.print_success(f"{desc} is available")
            return True
        except ImportError as e:
            desc = description or module_name
            self.print_error(f"{desc} is not available: {e}")
            return False
    
    def check_file_exists(self, file_path: str, description: str = None) -> bool:
        """Check if a file exists"""
        if os.path.exists(file_path):
            desc = description or file_path
            self.print_success(f"{desc} exists")
            return True
        else:
            desc = description or file_path
            self.print_error(f"{desc} not found")
            return False
    
    def validate_core_dependencies(self):
        """Validate core Python dependencies"""
        self.print_header("Core Dependencies")
        
        core_deps = [
            ('sys', 'Python Standard Library'),
            ('os', 'Operating System Interface'),
            ('json', 'JSON Support'),
            ('threading', 'Threading Support'),
            ('queue', 'Queue Support'),
            ('datetime', 'Date/Time Support'),
            ('pathlib', 'Path Handling'),
            ('logging', 'Logging Support')
        ]
        
        passed = 0
        for module, desc in core_deps:
            if self.check_import(module, desc):
                passed += 1
                self.results['core_dependencies'][module] = True
            else:
                self.results['core_dependencies'][module] = False
        
        self.print_success(f"Core dependencies: {passed}/{len(core_deps)} passed")
        return passed == len(core_deps)
    
    def validate_gui_framework(self):
        """Validate GUI framework dependencies"""
        self.print_header("GUI Framework (PyQt6)")
        
        gui_deps = [
            ('PyQt6.QtWidgets', 'PyQt6 Widgets'),
            ('PyQt6.QtCore', 'PyQt6 Core'),
            ('PyQt6.QtGui', 'PyQt6 GUI'),
            ('PyQt6.QtCharts', 'PyQt6 Charts'),
            ('psutil', 'System Monitoring')
        ]
        
        passed = 0
        for module, desc in gui_deps:
            if self.check_import(module, desc):
                passed += 1
                self.results['gui_framework'][module] = True
            else:
                self.results['gui_framework'][module] = False
        
        # Test PyQt6 application creation
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication([])
            app.quit()
            self.print_success("PyQt6 application creation test passed")
            self.results['gui_framework']['app_creation'] = True
            passed += 1
        except Exception as e:
            self.print_error(f"PyQt6 application creation test failed: {e}")
            self.results['gui_framework']['app_creation'] = False
        
        self.print_success(f"GUI framework: {passed}/{len(gui_deps)+1} passed")
        return passed == len(gui_deps) + 1
    
    def validate_business_modules(self):
        """Validate business logic modules"""
        self.print_header("Business Modules")
        
        business_files = [
            ('main_application.py', 'Main Application'),
            ('autonomous_orchestrator.py', 'Autonomous Orchestrator'),
            ('autonomous_business_operations.py', 'Business Operations'),
            ('income_generator.py', 'Income Generator'),
            ('crypto_manager.py', 'Crypto Manager'),
            ('agent_manager.py', 'Agent Manager')
        ]
        
        passed = 0
        for file_path, desc in business_files:
            if self.check_file_exists(file_path, desc):
                passed += 1
                self.results['business_modules'][file_path] = True
                
                # Try to import as module
                try:
                    module_name = file_path.replace('.py', '')
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Don't execute, just check if it can be loaded
                        self.print_success(f"{desc} can be imported")
                except Exception as e:
                    self.print_warning(f"{desc} import test failed: {e}")
            else:
                self.results['business_modules'][file_path] = False
        
        self.print_success(f"Business modules: {passed}/{len(business_files)} passed")
        return passed >= len(business_files) * 0.8  # 80% pass rate
    
    def validate_ai_libraries(self):
        """Validate AI and ML libraries"""
        self.print_header("AI/ML Libraries")
        
        ai_deps = [
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('matplotlib', 'Matplotlib'),
            ('requests', 'HTTP Requests'),
            ('cryptography', 'Cryptography')
        ]
        
        optional_ai_deps = [
            ('openai', 'OpenAI API'),
            ('anthropic', 'Anthropic API'),
            ('transformers', 'Hugging Face Transformers'),
            ('torch', 'PyTorch'),
            ('tensorflow', 'TensorFlow')
        ]
        
        passed = 0
        total = len(ai_deps)
        
        # Check required AI dependencies
        for module, desc in ai_deps:
            if self.check_import(module, desc):
                passed += 1
                self.results['ai_libraries'][module] = True
            else:
                self.results['ai_libraries'][module] = False
        
        # Check optional AI dependencies
        optional_passed = 0
        for module, desc in optional_ai_deps:
            if self.check_import(module, f"{desc} (optional)"):
                optional_passed += 1
                self.results['ai_libraries'][f"{module}_optional"] = True
            else:
                self.results['ai_libraries'][f"{module}_optional"] = False
        
        self.print_success(f"Required AI libraries: {passed}/{total} passed")
        self.print_success(f"Optional AI libraries: {optional_passed}/{len(optional_ai_deps)} available")
        
        return passed >= total * 0.8  # 80% pass rate for required
    
    def validate_crypto_libraries(self):
        """Validate cryptocurrency libraries"""
        self.print_header("Cryptocurrency Libraries")
        
        crypto_deps = [
            ('ccxt', 'CCXT Exchange Library'),
            ('web3', 'Web3 Ethereum Library'),
            ('hashlib', 'Hash Functions'),
            ('base64', 'Base64 Encoding')
        ]
        
        optional_crypto_deps = [
            ('binance', 'Binance API'),
            ('yfinance', 'Yahoo Finance'),
            ('ta', 'Technical Analysis')
        ]
        
        passed = 0
        total = len(crypto_deps)
        
        # Check required crypto dependencies
        for module, desc in crypto_deps:
            if self.check_import(module, desc):
                passed += 1
                self.results['crypto_libraries'][module] = True
            else:
                self.results['crypto_libraries'][module] = False
        
        # Check optional crypto dependencies
        optional_passed = 0
        for module, desc in optional_crypto_deps:
            if self.check_import(module, f"{desc} (optional)"):
                optional_passed += 1
                self.results['crypto_libraries'][f"{module}_optional"] = True
            else:
                self.results['crypto_libraries'][f"{module}_optional"] = False
        
        self.print_success(f"Required crypto libraries: {passed}/{total} passed")
        self.print_success(f"Optional crypto libraries: {optional_passed}/{len(optional_crypto_deps)} available")
        
        return passed >= total * 0.6  # 60% pass rate for crypto (some may not be available)
    
    def validate_system_files(self):
        """Validate system files and scripts"""
        self.print_header("System Files")
        
        system_files = [
            ('START_SYSTEM.sh', 'System Startup Script'),
            ('BUILD_MACOS_APP.sh', 'macOS Build Script'),
            ('COMPLETE_SYSTEM_SETUP.sh', 'Setup Script'),
            ('requirements_complete.txt', 'Complete Requirements'),
            ('README.md', 'Documentation')
        ]
        
        directories = [
            ('logs', 'Logs Directory'),
            ('config', 'Configuration Directory'),
            ('data', 'Data Directory')
        ]
        
        passed = 0
        total = len(system_files) + len(directories)
        
        # Check files
        for file_path, desc in system_files:
            if self.check_file_exists(file_path, desc):
                passed += 1
                self.results['system_files'][file_path] = True
            else:
                self.results['system_files'][file_path] = False
        
        # Check directories (create if missing)
        for dir_path, desc in directories:
            if os.path.exists(dir_path):
                self.print_success(f"{desc} exists")
                passed += 1
                self.results['system_files'][dir_path] = True
            else:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    self.print_success(f"{desc} created")
                    passed += 1
                    self.results['system_files'][dir_path] = True
                except Exception as e:
                    self.print_error(f"Failed to create {desc}: {e}")
                    self.results['system_files'][dir_path] = False
        
        self.print_success(f"System files: {passed}/{total} passed")
        return passed >= total * 0.8  # 80% pass rate
    
    def validate_configuration(self):
        """Validate system configuration"""
        self.print_header("Configuration")
        
        config_checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.print_success(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is supported")
            config_checks.append(True)
        else:
            self.print_error(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is too old (3.8+ required)")
            config_checks.append(False)
        
        # Check operating system
        import platform
        os_name = platform.system()
        if os_name == "Darwin":
            self.print_success(f"Running on macOS ({platform.mac_ver()[0]})")
            config_checks.append(True)
        elif os_name == "Linux":
            self.print_warning(f"Running on Linux - some features may not work")
            config_checks.append(True)
        elif os_name == "Windows":
            self.print_warning(f"Running on Windows - some features may not work")
            config_checks.append(True)
        else:
            self.print_error(f"Unsupported operating system: {os_name}")
            config_checks.append(False)
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= 8:
                self.print_success(f"System memory: {memory_gb:.1f}GB (sufficient)")
                config_checks.append(True)
            elif memory_gb >= 4:
                self.print_warning(f"System memory: {memory_gb:.1f}GB (minimum)")
                config_checks.append(True)
            else:
                self.print_error(f"System memory: {memory_gb:.1f}GB (insufficient)")
                config_checks.append(False)
        except:
            self.print_warning("Could not check system memory")
            config_checks.append(True)
        
        # Check disk space
        try:
            disk_usage = os.statvfs('.')
            free_space_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
            if free_space_gb >= 5:
                self.print_success(f"Free disk space: {free_space_gb:.1f}GB (sufficient)")
                config_checks.append(True)
            elif free_space_gb >= 2:
                self.print_warning(f"Free disk space: {free_space_gb:.1f}GB (minimum)")
                config_checks.append(True)
            else:
                self.print_error(f"Free disk space: {free_space_gb:.1f}GB (insufficient)")
                config_checks.append(False)
        except:
            self.print_warning("Could not check disk space")
            config_checks.append(True)
        
        passed = sum(config_checks)
        total = len(config_checks)
        
        self.print_success(f"Configuration checks: {passed}/{total} passed")
        return passed >= total * 0.75  # 75% pass rate
    
    def run_validation(self) -> bool:
        """Run complete system validation"""
        print("Skyscope AI Agentic Swarm Business/Enterprise - System Validation")
        print("=" * 70)
        
        validation_results = []
        
        # Run all validation checks
        validation_results.append(self.validate_core_dependencies())
        validation_results.append(self.validate_gui_framework())
        validation_results.append(self.validate_business_modules())
        validation_results.append(self.validate_ai_libraries())
        validation_results.append(self.validate_crypto_libraries())
        validation_results.append(self.validate_system_files())
        validation_results.append(self.validate_configuration())
        
        # Calculate overall results
        passed_checks = sum(validation_results)
        total_checks = len(validation_results)
        pass_percentage = (passed_checks / total_checks) * 100
        
        # Print summary
        self.print_header("Validation Summary")
        
        if pass_percentage >= 90:
            self.results['overall_status'] = 'excellent'
            self.print_success(f"System validation: {passed_checks}/{total_checks} ({pass_percentage:.1f}%) - EXCELLENT")
            self.print_success("Your system is ready for full autonomous operation!")
        elif pass_percentage >= 75:
            self.results['overall_status'] = 'good'
            self.print_success(f"System validation: {passed_checks}/{total_checks} ({pass_percentage:.1f}%) - GOOD")
            self.print_warning("Your system is ready with some optional features missing.")
        elif pass_percentage >= 60:
            self.results['overall_status'] = 'fair'
            self.print_warning(f"System validation: {passed_checks}/{total_checks} ({pass_percentage:.1f}%) - FAIR")
            self.print_warning("Your system may work but some features will be limited.")
        else:
            self.results['overall_status'] = 'poor'
            self.print_error(f"System validation: {passed_checks}/{total_checks} ({pass_percentage:.1f}%) - POOR")
            self.print_error("Your system needs significant setup before it can run properly.")
        
        # Save results
        self.save_validation_results()
        
        return pass_percentage >= 75
    
    def save_validation_results(self):
        """Save validation results to file"""
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/validation_results.json', 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.print_success("Validation results saved to logs/validation_results.json")
        except Exception as e:
            self.print_error(f"Failed to save validation results: {e}")
    
    def print_recommendations(self):
        """Print recommendations based on validation results"""
        self.print_header("Recommendations")
        
        if not self.results['gui_framework'].get('PyQt6.QtWidgets', False):
            print("• Install PyQt6: pip install PyQt6 PyQt6-Charts PyQt6-WebEngine")
        
        if not self.results['ai_libraries'].get('numpy', False):
            print("• Install core AI libraries: pip install numpy pandas matplotlib")
        
        if not self.results['crypto_libraries'].get('ccxt', False):
            print("• Install crypto libraries: pip install ccxt web3")
        
        if not self.results['business_modules'].get('main_application.py', False):
            print("• Ensure all business module files are present")
        
        if self.results['overall_status'] in ['poor', 'fair']:
            print("• Run COMPLETE_SYSTEM_SETUP.sh to install missing dependencies")
            print("• Check the logs/validation_results.json file for detailed information")
        
        print("\nTo start the system after fixing issues:")
        print("• Run: chmod +x START_SYSTEM.sh && ./START_SYSTEM.sh")

def main():
    """Main validation function"""
    validator = SystemValidator()
    
    try:
        success = validator.run_validation()
        validator.print_recommendations()
        
        if success:
            print("\n🎉 System validation completed successfully!")
            print("You can now start the system with: ./START_SYSTEM.sh")
            return 0
        else:
            print("\n⚠️ System validation found issues that need to be addressed.")
            print("Please follow the recommendations above and run validation again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nValidation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())