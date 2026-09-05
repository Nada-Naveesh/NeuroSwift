from pathlib import Path

from setuptools import find_packages, setup

readme = Path("README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else "NeuroSwift"

setup(
    name="neuroswift",
    version="0.1.0",
    description="Motor imagery classification from EEG signals using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(include=["models", "models.*", "preprocessing", "preprocessing.*", "training", "training.*", "src", "src.*", "demo", "demo.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "mne",
        "torch",
        "scikit-learn",
        "streamlit",
        "plotly",
        "tqdm",
        "pyyaml",
    ],
)
