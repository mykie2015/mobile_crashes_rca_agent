"""Setup script for Mobile Crashes RCA Agent."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="mobile-crashes-rca-agent",
    version="1.0.0",
    description="AI-powered mobile application crash analysis agent using LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mobile RCA Team",
    author_email="team@example.com",
    url="https://github.com/example/mobile-crashes-rca-agent",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
        "data": ["prompts/*", "cache/*"],
    },
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "mobile-rca=src.mobile_crash_agent:main",
        ],
    },
    keywords="ai, llm, crash-analysis, mobile, langraph, openai",
) 