"""
Install with
pip install --extra-index-url https://gitlab.lrz.de/api/v4/projects/54930/packages/pypi mod_prediction
"""

import setuptools
import subprocess


def git(*args):
    return subprocess.check_output(["git"] + list(args))


# get latest tag
latest = git("describe", "--tags").decode().strip()
latest = latest.split("-")[0]

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


long_description = """
This repository provides a LSTM neural network for vehicle trajectory prediction with uncertainties.
The provided models and interfaces are optimized for the usage with CommonRoad.

Deployment in Motion Planning Framework:\n
1. Import the Prediction class (from pred import Prediction) \n
2. Intialize the Prediction class with a scenario (predictor = Prediction(scenario)) \n
3. Loop over the step function and provide time step and obstacle ID (Prediction.step(time_step, ost_id))

For further information see the Readme here:
https://gitlab.lrz.de/ga38hip/pred/-/blob/master/README.md
"""


setuptools.setup(
    name="mod_prediction",
    version=latest,
    author="Maximilian Geisslinger, Phillip Karle",
    author_email="maximilian.geisslinger@tum.de, phillip.karle@tum.de",
    description="Prediction module for CommonRoad",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.lrz.de/motionplanning1/mod_prediction",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
