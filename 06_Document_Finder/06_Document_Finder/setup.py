"""Setup script for Document Search Engine."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
try:
    long_description = (this_directory / "README.md").read_text()
except FileNotFoundError:
    long_description = "A production-grade document search engine supporting PDF, Word, and Markdown files."

# Read requirements
requirements = []
try:
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'PyPDF2>=3.0.0',
        'python-docx>=1.1.0',
        'markdown>=3.5.0',
        'pyyaml>=6.0.0'
    ]

setup(
    name="document-search-engine",
    version="1.0.0",
    author="Databricks",
    description="A production-grade document search engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/databricks/document-search-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'docsearch=src.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)