"""
setup.py — Makes the project installable as a package.
Run: pip install -e .
This allows `from src.utils import ...` imports to work from any directory.
"""

from setuptools import setup, find_packages

setup(
    name="hallucinations-in-ai",
    version="1.0.0",
    description="U2UInnovate Project 22 — Hallucination Detection in LLMs",
    packages=find_packages(exclude=["venv", "*.egg-info", "tests"]),
    python_requires=">=3.11",
)
