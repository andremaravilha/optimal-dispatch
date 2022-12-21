import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimaldipatch",
    version="1.0.0",
    author="Maria Izabel C. Santos, André L. Maravilha",
    author_email="izabel.mics@gmail.com, andre.maravilha@cefetmg.br",
    description="Algorithms for optimal dispatch in a microgrid",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andremaravilha/optimal-dispatch",
    packages=setuptools.find_packages(),
    requires=["numpy", "matplotlib", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
