from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# remove local package install '-e .' if present
requirements = [r for r in requirements if not r.startswith('-e')]

setup(
    name='shinobi_fmri',
    packages=find_packages(),
    version='0.1.0',
    description='fMRI analysis of the cneuromod.shinobi dataset',
    author='Yann Harel',
    license='MIT',
    install_requires=requirements,
)
