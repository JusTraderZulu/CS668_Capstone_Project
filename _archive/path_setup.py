"""
Path Setup Module

This module configures the Python path to include the necessary directories
for importing modules within the DQL Trading Framework.
"""
import os
import sys

def add_project_root_to_path():
    """
    Add the project root directory to Python's sys.path to allow imports
    from any part of the project.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add root to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # Print for debugging
    # print(f"Added {project_root} to Python path")
    # print(f"Current Python path: {sys.path}")
    
# Add paths when this module is imported
add_project_root_to_path() 