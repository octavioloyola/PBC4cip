import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-rs_ernesto", # Replace with your own username
    version="0.0.1",
    author="Ernesto Ramírez Sáyago, Miguel Angel Medina Pérez",
    author_email="A00513925@itesm.mx, migue@tec.mx",
    description="PBC4cip classifier for class imbalence problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/octavioloyola/PBC4cip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
