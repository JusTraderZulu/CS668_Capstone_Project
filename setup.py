from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dql_trading",
    version="0.1.0",
    author="Justin Borneo",
    author_email="your.email@example.com",
    description="A Deep Q-Learning framework for algorithmic trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DQL_agent",
    packages=find_packages(include=["dql_trading", "dql_trading.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dql-trading=dql_trading.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dql_trading": ["data/*.csv"],
    },
)
