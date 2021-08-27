from setuptools import setup, find_packages
from typing import Dict

setup(
    name="zsl-kg",
    version="0.0.1",
    url="",
    author="Nihal V. Nayak, Stephen H. Bach",
    author_email="nnayak2@cs.brown.edu",
    description="Zero-shot learning with Common Sense Knowledge Graphs",
    packages=find_packages(),
    install_requires=[
        "allennlp==0.9.0",
        "torch==1.6.0",
        "torchvision==0.7.0",
        "overrides==3.1.0",
        "pandas==1.1.3",
        "numpy>=1.16.5",
    ],
)
