import os
import subprocess
import sys

def run_app():
    """
    Entry point to run the Streamlit app.
    """
    # Get the path to app.py relative to this file
    # This assumes app.py is in the root and retina_risk is a subfolder
    base_dir = os.path.dirname(os.path.dirname(__file__))
    app_path = os.path.join(base_dir, "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
        
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    run_app()
