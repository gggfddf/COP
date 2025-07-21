#!/usr/bin/env python3
"""
📋 FILE LIST GENERATOR - Gold AI System
Shows exactly what files you have and where they should go
"""

import os
import sys

def show_complete_file_structure():
    """Show the complete file structure for the Gold AI System"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                📋 GOLD AI SYSTEM - FILE LIST                ║
║              WHAT YOU NEED ON YOUR PC                       ║
╚══════════════════════════════════════════════════════════════╝

🏠 CREATE THIS FOLDER STRUCTURE ON YOUR PC:
""")
    
    files_structure = {
        "📁 Gold-AI-System/ (Main folder)": [
            "execute_gold_ai.py",
            "main.py", 
            "full_system_execution.py",
            "simple_gold_ai_demo.py",
            "requirements.txt",
            "README.md",
            "SYSTEM_STATUS.md",
            "DEPLOYMENT_GUIDE.md",
            "PC_SETUP_GUIDE.md"
        ],
        "📁 src/data/": [
            "data_collector.py"
        ],
        "📁 src/features/": [
            "feature_engineer.py"
        ],
        "📁 src/models/": [
            "hybrid_model.py"
        ],
        "📁 src/patterns/": [
            "pattern_discovery.py"
        ],
        "📁 src/prediction/": [
            "prediction_system.py"
        ],
        "📁 config/": [
            "config.yaml"
        ]
    }
    
    total_files = 0
    for folder, files in files_structure.items():
        print(f"\n{folder}")
        for file in files:
            print(f"   ├── {file}")
            total_files += 1
    
    print(f"\n📊 TOTAL FILES NEEDED: {total_files}")
    
    print("""
📝 FOLDERS THAT WILL BE CREATED AUTOMATICALLY:
   📁 output/          (prediction results)
   📁 logs/            (system logs)
   📁 models/          (trained AI models)
   📁 complete_results/ (full analysis results)

🚀 QUICK START COMMANDS:
   1. python simple_gold_ai_demo.py      # Test the system
   2. python execute_gold_ai.py          # Your main script
   3. python full_system_execution.py    # Complete analysis

💻 REQUIRED SOFTWARE:
   ✅ Python 3.8 or newer
   ✅ Internet connection (for market data)
   ✅ 2GB free disk space
   ✅ Windows/Mac/Linux (any OS)
""")

def check_current_files():
    """Check what files are currently available"""
    
    print("\n" + "="*60)
    print("🔍 CHECKING CURRENT FILES IN THIS DIRECTORY:")
    print("="*60)
    
    required_files = [
        "execute_gold_ai.py",
        "main.py", 
        "full_system_execution.py",
        "simple_gold_ai_demo.py",
        "requirements.txt",
        "src/data/data_collector.py",
        "src/features/feature_engineer.py",
        "src/models/hybrid_model.py",
        "src/patterns/pattern_discovery.py",
        "src/prediction/prediction_system.py",
        "config/config.yaml"
    ]
    
    found_files = []
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            found_files.append(file_path)
            print(f"✅ FOUND: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ MISSING: {file_path}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Found: {len(found_files)}/{len(required_files)} files")
    print(f"   ❌ Missing: {len(missing_files)} files")
    
    if len(missing_files) == 0:
        print(f"\n🎉 ALL FILES PRESENT! Ready to run the system!")
        print(f"   Run: python simple_gold_ai_demo.py")
    else:
        print(f"\n⚠️  MISSING FILES - You need to add these files:")
        for file in missing_files:
            print(f"   📄 {file}")

def show_run_instructions():
    """Show how to run the system"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🚀 HOW TO RUN                            ║
╚══════════════════════════════════════════════════════════════╝

📋 STEP-BY-STEP INSTRUCTIONS:

1️⃣ INSTALL PYTHON:
   • Go to python.org/downloads
   • Download Python 3.8 or newer
   • Check "Add Python to PATH" during install

2️⃣ CREATE FOLDER:
   • Create folder: Gold-AI-System
   • Copy all files to this folder

3️⃣ INSTALL PACKAGES:
   • Open Command Prompt/Terminal
   • Go to your Gold-AI-System folder
   • Run: pip install -r requirements.txt

4️⃣ RUN THE SYSTEM:
   • Start with: python simple_gold_ai_demo.py
   • Then try: python execute_gold_ai.py
   • Full system: python full_system_execution.py

🎯 WHAT EACH SCRIPT DOES:

📄 simple_gold_ai_demo.py
   • Quick demo (30 seconds)
   • Tests all components
   • Shows sample predictions
   • No real data needed

📄 execute_gold_ai.py  
   • Your main execution script
   • Full AI analysis
   • Real market data
   • Complete predictions

📄 full_system_execution.py
   • Complete multi-timeframe analysis
   • 6 timeframes (1m to 1d)
   • 1,400+ predictions
   • Comprehensive results

📄 main.py
   • Master orchestrator
   • Custom configurations
   • Advanced options

🔧 TROUBLESHOOTING:
   • Use python3 instead of python (Mac/Linux)
   • Run as Administrator (Windows)
   • Check internet connection
   • Verify all files are present
""")

if __name__ == "__main__":
    show_complete_file_structure()
    check_current_files()
    show_run_instructions()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ READY TO GO!                          ║
║                                                              ║
║  Copy all files to your PC and run:                         ║
║  python simple_gold_ai_demo.py                              ║
╚══════════════════════════════════════════════════════════════╝
""")