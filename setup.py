from setuptools import setup, find_packages


setup(
    name="EnergyAction",
    version="0.1.0",
    packages=find_packages(where="."),   # or specify subfolders
    package_dir={"": "."},               # interpret current dir as top-level
)
