import pytest
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_tests():
    # Run pytest with verbosity
    pytest.main(['-v', os.path.dirname(__file__)])

if __name__ == '__main__':
    run_tests() 