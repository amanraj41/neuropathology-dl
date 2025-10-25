"""
Quick Validation Script

This script validates the code structure and syntax without requiring
all dependencies to be installed. It's useful for CI/CD or quick checks.

Usage:
    python validate.py
"""

import ast
import os
import sys
from pathlib import Path


def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def validate_project():
    """Validate the project structure and code."""
    print("="*70)
    print("PROJECT VALIDATION")
    print("="*70)
    
    # Expected files
    expected_files = [
        'app.py',
        'train.py',
        'demo.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/neuropathology_model.py',
        'src/data/__init__.py',
        'src/data/data_loader.py',
        'src/utils/__init__.py',
        'src/utils/helpers.py',
    ]
    
    print("\nChecking project structure...")
    all_exist = True
    for filepath in expected_files:
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n✗ Some expected files are missing!")
        return False
    
    print("\n✓ All expected files present")
    
    # Check Python files syntax
    print("\nValidating Python syntax...")
    python_files = [f for f in expected_files if f.endswith('.py')]
    
    all_valid = True
    for filepath in python_files:
        valid, error = check_syntax(filepath)
        status = "✓" if valid else "✗"
        print(f"  {status} {filepath}")
        if not valid:
            print(f"      Error: {error}")
            all_valid = False
    
    if not all_valid:
        print("\n✗ Some files have syntax errors!")
        return False
    
    print("\n✓ All Python files have valid syntax")
    
    # Check directory structure
    print("\nChecking directory structure...")
    expected_dirs = ['src', 'src/models', 'src/data', 'src/utils', 'models']
    
    for dirpath in expected_dirs:
        exists = os.path.exists(dirpath) and os.path.isdir(dirpath)
        status = "✓" if exists else "✗"
        print(f"  {status} {dirpath}/")
    
    # Count lines of code
    print("\nProject statistics...")
    total_lines = 0
    total_docstrings = 0
    
    for filepath in python_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines += len(lines)
            
            # Count docstring lines (rough estimate)
            in_docstring = False
            for line in lines:
                stripped = line.strip()
                if '"""' in stripped or "'''" in stripped:
                    in_docstring = not in_docstring
                if in_docstring or '"""' in stripped or "'''" in stripped:
                    total_docstrings += 1
    
    print(f"  Total lines of code: {total_lines}")
    print(f"  Documentation lines: {total_docstrings}")
    print(f"  Documentation ratio: {total_docstrings/total_lines*100:.1f}%")
    
    # Check README
    print("\nChecking documentation...")
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
        readme_lines = len(readme_content.split('\n'))
        print(f"  README lines: {readme_lines}")
        
        # Check for key sections
        sections = [
            'Overview', 'Features', 'Installation', 'Usage',
            'Model Architecture', 'Training', 'Deep Learning'
        ]
        for section in sections:
            if section in readme_content:
                print(f"  ✓ Section found: {section}")
            else:
                print(f"  ⚠ Section might be missing: {section}")
    
    print("\n" + "="*70)
    print("✓ VALIDATION COMPLETE")
    print("="*70)
    print("\nThe project structure is valid and ready for use.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run demo: python demo.py")
    print("3. Train model: python train.py --data_dir /path/to/data")
    print("4. Launch app: streamlit run app.py")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = validate_project()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Validation failed with error: {str(e)}")
        sys.exit(1)
