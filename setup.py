from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="autoencoders",  # Replace with your project name
    version="0.1.0",  # Semantic versioning (Major.Minor.Patch)
    author="Ali",  # Replace with your name
    author_email="alireza202@gmail.com",
    description="A repo for testing out various autoencoders",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NeuralPensieve/autoencoders",  # Replace with your repo URL
    packages=find_packages(),  # Automatically discover sub-packages
    install_requires=requirements,  # Load from requirements.txt
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",  # Adjust based on your target Python version
        "License :: OSI Approved :: MIT License",  # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version
)