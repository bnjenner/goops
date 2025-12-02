from setuptools import setup, find_packages

setup(
    name="goops",
    version="0.0.1",
    description="Partitions sequences through  motifs discovery using the goops algorithm.",
    author="B. N. Jenner, A. P. Lee, E. X. Markert",
    python_requires=">=3.8",
    packages=find_packages(include=["goops", "goops.*"]),
    entry_points={
        "console_scripts": [
            "goops = goops.goops:main",
        ],
    },
)
