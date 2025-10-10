#!/usr/bin/env python3
"""
Run the complete SRL-to-RDF pipeline using virtual environments
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_in_venv(venv_name, script_path, args):
    """Run a script in a virtual environment"""
    if platform.system() == "Windows":
        python_cmd = f"{venv_name}\\Scripts\\python"
    else:
        python_cmd = f"{venv_name}/bin/python"
    
    cmd = [python_cmd, script_path] + args
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_path}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 SRL-to-RDF Pipeline")
    print("=" * 40)
    
    # Check if virtual environments exist
    venvs = [".srl_env", ".orchestrator_env", ".rel_env"]
    for venv in venvs:
        if not os.path.exists(venv):
            print(f"❌ Virtual environment {venv} not found!")
            print("Run: python setup_simple.py first")
            return 1
    
    # Create output directories
    output_dirs = ["outputs", "outputs/srl", "outputs/rel", "outputs/rdf"]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create sample input if it doesn't exist
    if not os.path.exists("sample_input.txt"):
        sample_text = """Steve Jobs founded Apple in California. Apple designed the iPhone and the MacBook. Tim Cook announced a new iPhone in San Francisco in 2015. Albert Einstein was born in Germany in 1879; he received the Nobel Prize in Physics in 1921 for the photoelectric effect, and he died in New Jersey in 1955. Lionel Messi scored three goals for FC Barcelona at Camp Nou; commentators called his performance historic."""
        
        with open("sample_input.txt", "w", encoding="utf-8") as f:
            f.write(sample_text)
        print("📝 Created sample_input.txt")
    
    # Step 1: SRL
    print("\n🔍 Step 1/3: Semantic Role Labeling...")
    if not run_in_venv(".srl_env", "modules/framesrl/framesrl_runner.py", 
                      ["--infile", "sample_input.txt", 
                       "--outfile", "outputs/srl/frames.json", 
                       "--model", "base"]):
        return 1
    
    # Step 2: Entity Linking
    print("\n🔗 Step 2/3: Entity Linking...")
    if not run_in_venv(".rel_env", "modules/rel_linker/rel_runner_fixed.py", 
                      ["--infile", "sample_input.txt", 
                       "--outfile", "outputs/rel/entities.json",
                       "--confidence", "0.38"]):
        return 1
    
    # Step 3: RDF Conversion
    print("\n🔄 Step 3/3: RDF Conversion...")
    rdf_args = ["--frames", "outputs/srl/frames.json", 
                "--entities", "outputs/rel/entities.json", 
                "--outfile", "outputs/rdf/rdf_output.ttl"]
    
    # Add DeepInfra parameters
    rdf_args.extend(["--deepinfra-model", "meta-llama/Llama-2-70b-chat-hf"])
    
    # Get DeepInfra API key from environment or use default
    deepinfra_key = os.getenv("DEEPINFRA_API_KEY", "WJkNzU3cHwGGC6d5nGrGGCoFF9qIW8li")
    rdf_args.extend(["--deepinfra-api-key", deepinfra_key])
    print("🤖 Using DeepInfra for pronoun coreference resolution")
    
    if not run_in_venv(".orchestrator_env", "modules/orchestrator/rdfify_improved.py", rdf_args):
        return 1
    
    print("\n🎉 Pipeline completed successfully!")
    print("\n📁 Output files:")
    print("  - Frames: outputs/srl/frames.json")
    print("  - Entities: outputs/rel/entities.json") 
    print("  - RDF: outputs/rdf/rdf_output.ttl")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
