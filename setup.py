# ~/ProRAG/setup.py
from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name="prorag",
    version="0.1.0",
    description="ProRAG: Progressive RAG Training Framework",
    author="AISC User",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=parse_requirements("requirements.txt"),
)