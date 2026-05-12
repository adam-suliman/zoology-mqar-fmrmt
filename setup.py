from setuptools import find_packages, setup


_REQUIRED = [
    "numpy",
    "einops",
    "tqdm",
    "click",
    "pydantic",
    "pandas",
    "wandb",
    "comet_ml",
    "rotary-embedding-torch",
    "einx",
    "transformers",
    "PyYAML",
]

_OPTIONAL = {
    "analysis": [
        "pandas",
        "seaborn",
        "matplotlib",
    ],
    "extra":[
        "rich", 
        "ray",
    ],
    "extras": [
        "rich",
        "ray",
    ],
    "test": [
        "pytest",
    ],
    "cuda_mixers": [
        "flash-linear-attention",
        "causal_conv1d",
    ],
}

setup(
    name="zoology", 
    version="0.0.1",
    description="",
    packages=find_packages(),  
    install_requires=_REQUIRED,
    extras_require=_OPTIONAL,
    entry_points={
        'console_scripts': ['zg=zoology.cli:cli'],
    },
)
