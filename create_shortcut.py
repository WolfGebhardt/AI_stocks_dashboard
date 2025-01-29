import os
import sys
import platform
import winreg  # For Windows registry access

def get_windows_desktop_path():
    """Get Windows desktop path from registry"""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
            desktop_path = winreg.QueryValueEx(key, "Desktop")[0]
            return desktop_path
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Desktop")

def create_desktop_shortcut():
    try:
        # Get the desktop path based on OS
        if platform.system() == "Windows":
            desktop = get_windows_desktop_path()
        else:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # Ensure desktop directory exists
        if not os.path.exists(desktop):
            print(f"Desktop path not found at: {desktop}")
            desktop = os.getcwd()  # Fallback to current directory
            print(f"Falling back to current directory: {desktop}")

        # Get absolute paths
        current_dir = os.path.abspath(os.getcwd())
        venv_path = os.path.abspath(os.path.join(current_dir, "venv"))
        script_path = os.path.abspath(os.path.join(current_dir, "ai_stocks_dashboard.py"))

        # Print paths for debugging
        print(f"Current directory: {current_dir}")
        print(f"Virtual environment path: {venv_path}")
        print(f"Script path: {script_path}")
        
        if platform.system() == "Windows":
            shortcut_path = os.path.join(desktop, "AI_Stocks_Dashboard.bat")
            
            bat_content = f"""@echo off
cd /d "{current_dir}"
call "{os.path.join(venv_path, 'Scripts', 'activate.bat')}"
streamlit run "{script_path}"
pause
"""
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(shortcut_path), exist_ok=True)
            
            try:
                with open(shortcut_path, "w") as f:
                    f.write(bat_content)
                print(f"Successfully created shortcut at: {shortcut_path}")
            except Exception as e:
                print(f"Error writing shortcut file: {e}")
                
                # Fallback to current directory if desktop fails
                shortcut_path = os.path.join(current_dir, "AI_Stocks_Dashboard.bat")
                with open(shortcut_path, "w") as f:
                    f.write(bat_content)
                print(f"Created shortcut in current directory instead: {shortcut_path}")
        else:
            shortcut_path = os.path.join(desktop, "AI_Stocks_Dashboard.sh")
            
            sh_content = f"""#!/bin/bash
cd "{current_dir}"
source "{os.path.join(venv_path, 'bin', 'activate')}"
streamlit run "{script_path}"
"""
            try:
                with open(shortcut_path, "w") as f:
                    f.write(sh_content)
                os.chmod(shortcut_path, 0o755)
                print(f"Successfully created shortcut at: {shortcut_path}")
            except Exception as e:
                print(f"Error creating shortcut: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_desktop_shortcut()