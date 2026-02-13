from setuptools import setup, find_packages

setup(
    name="rul",
    version="0.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tensorflow",
        "torch",
        "pyyaml",
        "tqdm",
        "statsmodels",
        "lightgbm",
    ],
)
