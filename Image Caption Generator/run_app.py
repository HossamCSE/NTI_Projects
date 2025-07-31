#!/usr/bin/env python3
"""
Simple script to run the Streamlit app
"""
import os
import subprocess
import sys

def check_requirements():
    """Check if required model files exist"""
    required_files = [
        'best_caption_model.h5',
        'final_caption_model.h5',  # Alternative model file
        'tokenizer.pkl'
    ]
    
    model_exists = any(os.path.exists(f) for f in required_files[:2])
    tokenizer_exists = os.path.exists(required_files[2])
    
    if not model_exists:
        print("❌ Model file not found!")
        print("Please ensure one of these files exists:")
        print("  - best_caption_model.h5")
        print("  - final_caption_model.h5")
        return False
    
    if not tokenizer_exists:
        print("❌ Tokenizer file not found!")
        print("Please ensure 'tokenizer.pkl' exists in the current directory")
        return False
    
    print("✅ All required files found!")
    return True

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements!")
        return False

def run_streamlit():
    """Run the Streamlit app"""
    try:
        print("🚀 Starting Streamlit app...")
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    print("🖼️ Image Caption Generator Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n📝 Please ensure you have the trained model files in the current directory")
        sys.exit(1)
    
    # Install requirements if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        if not install_requirements():
            sys.exit(1)
    
    # Run the app
    print("\n🌐 The app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    run_streamlit()