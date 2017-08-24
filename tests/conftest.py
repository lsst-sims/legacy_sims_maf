import sys

def pytest_configure(config):
    sys._is_in_pytest = True
