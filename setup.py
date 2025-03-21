from setuptools import setup, find_packages

setup(
    name="octocli",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "pygithub>=2.1.1",
        "pymilvus>=2.3.4",
        "rich>=13.7.0",
        "radon>=6.0.1",
        "tree-sitter>=0.20.4",
        "pydantic>=2.5.2",
    ],
    entry_points={
        "console_scripts": [
            "octo=octocli.cli:app",
        ],
    },
) 