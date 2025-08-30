from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = 'projhotelres',
    version = '0.1',
    author='pranab',
    packages = find_packages(),
    install_requires = requirements
    )