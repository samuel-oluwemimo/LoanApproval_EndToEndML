from setuptools import find_packages, setup
from typing import List

EDOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """This function returns a list of requirements from a given file path."""
    requirements=[]
    with open(file_path) as f:
        requirements = f.read().splitlines()

        if EDOT in requirements:
            requirements.remove(EDOT)

    return requirements


setup(
    name='EE_Project',
    version='0.12.1',
    author='samuel',
    author_email='ayodejioluwemimo792@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)