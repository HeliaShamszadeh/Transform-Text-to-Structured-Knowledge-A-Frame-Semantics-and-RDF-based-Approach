#!/usr/bin/env python3
"""
Simple setup for SRL-to-RDF pipeline - bypasses pip upgrade issues
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Simple SRL-to-RDF Setup")
    print("=" * 40)
    
    # Check Python 3.10
    python_311_available = False
    try:
        result = subprocess.run(["py", "-3.11", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            python_311_available = True
            print("✅ Python 3.11 found")
    except:
        pass
    
    if not python_311_available:
        print("❌ Python 3.11 not found. Please install Python 3.11 first.")
        print("Download from: https://www.python.org/downloads/release/python-31011/")
        return 1
    
    # Create virtual environment for SRL
    print("\n📦 Creating .srl_env with Python 3.11...")
    if not run_command("py -3.11 -m venv .srl_env"):
        return 1
    
    # Create virtual environment for orchestrator
    print("\n📦 Creating .orchestrator_env...")
    if not run_command("python -m venv .orchestrator_env"):
        return 1
    
    # Create virtual environment for rel_linker
    print("\n📦 Creating .rel_env...")
    if not run_command("python -m venv .rel_env"):
        return 1
    
    # Install SRL dependencies
    print("\n🔧 Installing SRL dependencies...")
    if platform.system() == "Windows":
        srl_pip = ".srl_env\\Scripts\\pip"
        orch_pip = ".orchestrator_env\\Scripts\\pip"
        rel_pip = ".rel_env\\Scripts\\pip"
    else:
        srl_pip = ".srl_env/bin/pip"
        orch_pip = ".orchestrator_env/bin/pip"
        rel_pip = ".rel_env/bin/pip"
    
    # SRL packages
    srl_packages = ["nltk", "frame-semantic-transformer"]
    for package in srl_packages:
        print(f"Installing {package} in .srl_env...")
        if not run_command(f"{srl_pip} install {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Orchestrator packages
    print("Installing rdflib in .orchestrator_env...")
    if not run_command(f"{orch_pip} install rdflib>=7"):
        print("⚠️  Failed to install rdflib, continuing...")
    print("Installing requests in .orchestrator_env...")
    if not run_command(f"{orch_pip} install requests"):
        print("⚠️  Failed to install requests, continuing...")
    # print("Installing openai in .orchestrator_env...")
    # if not run_command(f"{orch_pip} install openai>=1.0.0"):
    #     print("⚠️  Failed to install openai, continuing...")
    print("Installing spacy in .orchestrator_env...")
    if not run_command(f"{orch_pip} install spacy"):
        print("⚠️  Failed to install spacy, continuing...")
    
    # REL packages
    print("Installing requests in .rel_env...")
    if not run_command(f"{rel_pip} install requests"):
        print("⚠️  Failed to install requests, continuing...")
    
    print("\n🎉 Setup complete!")
    print("\nTo run the pipeline:")
    print("python run_pipeline.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
