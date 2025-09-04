#!/usr/bin/env python3
"""
Deployment Script for Vietnamese AMR Semantic Parsing
====================================================

Easy deployment script for Hugging Face and Gradio.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install deployment requirements."""
    print("📦 Installing deployment requirements...")
    
    requirements = [
        "gradio",
        "huggingface_hub"
    ]
    
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
            print(f"✅ {req} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")
            return False
    
    return True

def push_to_huggingface(model_path, repo_name, token=None, private=False):
    """Push model to Hugging Face."""
    print(f"🚀 Pushing model to Hugging Face: {repo_name}")
    
    cmd = [
        sys.executable, "main.py", "push-model",
        "--model-path", model_path,
        "--repo-name", repo_name
    ]
    
    if token:
        cmd.extend(["--token", token])
    
    if private:
        cmd.append("--private")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Model pushed to: https://huggingface.co/{repo_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push model: {e}")
        return False

def launch_gradio(model_path=None, hf_model=None, port=7860, share=False):
    """Launch Gradio interface."""
    print("🌐 Launching Gradio interface...")
    
    cmd = [sys.executable, "main.py", "gradio"]
    
    if model_path:
        cmd.extend(["--model-path", model_path])
    
    if hf_model:
        cmd.extend(["--hf-model", hf_model])
    
    cmd.extend(["--port", str(port)])
    
    if share:
        cmd.append("--share")
    
    try:
        print(f"🎯 Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Gradio: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Gradio app stopped by user")
        return True

def create_huggingface_space(repo_name, token):
    """Create Hugging Face Space for deployment."""
    print(f"🏗️  Creating Hugging Face Space: {repo_name}")
    
    # Create app.py for HF Spaces
    app_content = f'''import gradio as gr
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our gradio app
from gradio_app import AMRGradioApp

# Create app with HF model
app = AMRGradioApp(hf_model_name="{repo_name}")
interface = app.create_interface()

# Launch for HF Spaces
if __name__ == "__main__":
    interface.launch()
'''
    
    # Create requirements.txt for HF Spaces
    requirements_content = '''transformers
torch
gradio
numpy
'''
    
    # Create README.md for HF Spaces
    readme_content = f'''---
title: Vietnamese AMR Parser
emoji: 🇻🇳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Vietnamese AMR Semantic Parsing

This is a Gradio app for Vietnamese Abstract Meaning Representation (AMR) semantic parsing.

## Usage

Enter Vietnamese text to get AMR representation.

## Model

This app uses the model: {repo_name}
'''
    
    print("📝 Creating Space files...")
    
    # Write files
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(app_content)
    
    with open("requirements_space.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    
    with open("README_space.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Space files created:")
    print("   - app.py (main app file)")
    print("   - requirements_space.txt (dependencies)")
    print("   - README_space.md (space description)")
    print()
    print("📋 Next steps:")
    print(f"1. Create a new Space at: https://huggingface.co/new-space")
    print(f"2. Upload these files to your Space repository")
    print(f"3. Your app will be available at: https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME")

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Vietnamese AMR Parser")
    
    subparsers = parser.add_subparsers(dest="command", help="Deployment commands")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install deployment requirements")
    
    # Push command
    push_parser = subparsers.add_parser("push", help="Push model to Hugging Face")
    push_parser.add_argument("--model-path", required=True, help="Path to trained model")
    push_parser.add_argument("--repo-name", required=True, help="HF repository name")
    push_parser.add_argument("--token", help="HF token")
    push_parser.add_argument("--private", action="store_true", help="Private repository")
    
    # Gradio command
    gradio_parser = subparsers.add_parser("gradio", help="Launch Gradio interface")
    gradio_parser.add_argument("--model-path", help="Path to local model")
    gradio_parser.add_argument("--hf-model", help="Hugging Face model name")
    gradio_parser.add_argument("--port", type=int, default=7860, help="Port")
    gradio_parser.add_argument("--share", action="store_true", help="Create public link")
    
    # Space command
    space_parser = subparsers.add_parser("space", help="Create HF Space files")
    space_parser.add_argument("--repo-name", required=True, help="HF model repository name")
    space_parser.add_argument("--token", help="HF token")
    
    # Full deployment
    deploy_parser = subparsers.add_parser("full", help="Full deployment (push + gradio)")
    deploy_parser.add_argument("--model-path", required=True, help="Path to trained model")
    deploy_parser.add_argument("--repo-name", required=True, help="HF repository name")
    deploy_parser.add_argument("--token", help="HF token")
    deploy_parser.add_argument("--private", action="store_true", help="Private repository")
    deploy_parser.add_argument("--port", type=int, default=7860, help="Port")
    deploy_parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🚀 Vietnamese AMR Deployment Tool")
    print("=" * 50)
    
    if args.command == "install":
        success = install_requirements()
        if success:
            print("✅ All requirements installed!")
        else:
            print("❌ Some requirements failed to install")
    
    elif args.command == "push":
        success = push_to_huggingface(
            args.model_path, 
            args.repo_name, 
            args.token, 
            args.private
        )
        if success:
            print("✅ Model pushed successfully!")
    
    elif args.command == "gradio":
        launch_gradio(
            args.model_path, 
            args.hf_model, 
            args.port, 
            args.share
        )
    
    elif args.command == "space":
        create_huggingface_space(args.repo_name, args.token)
    
    elif args.command == "full":
        # Install requirements
        print("Step 1: Installing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements")
            return
        
        # Push model
        print("\nStep 2: Pushing model to Hugging Face...")
        if not push_to_huggingface(args.model_path, args.repo_name, args.token, args.private):
            print("❌ Failed to push model")
            return
        
        # Create space files
        print("\nStep 3: Creating Space files...")
        create_huggingface_space(args.repo_name, args.token)
        
        # Launch Gradio
        print("\nStep 4: Launching Gradio interface...")
        launch_gradio(None, args.repo_name, args.port, args.share)

if __name__ == "__main__":
    main()
