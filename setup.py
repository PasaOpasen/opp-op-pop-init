
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="OppOpPopInit", 
    version="1.0.1",
    author="Demetry Pascal",
    author_email="qtckpuhdsa@gmail.com",
    maintainer = ['Demetry Pascal'],
    description="PyPI package containing opposition learning operators and population initializers for evolutionary algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PasaOpasen/opp-op-pop-init",
    keywords=['opposition-based-learning', 'optimization', 'evolutionary algorithms', 'fast', 'easy', 'evolution', 'generator'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['numpy']
    
    )





