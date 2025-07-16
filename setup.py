from setuptools import setup, find_packages

setup(
    name="theorygamesdl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "statsmodels",
        "tqdm",
    ],
    author="TheoryGamesDL Team",
    author_email="example@example.com",
    description="A library for game theory simulations with deep learning approaches",
    keywords="game theory, deep learning, reinforcement learning",
    python_requires=">=3.6",
)