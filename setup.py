from setuptools import setup, find_packages

setup(
    name="Credit_Risk_Modelling",
    version="0.0.1",
    author="Aryan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)