# Contributor documentation

## Contributing to the grain project documentation

## Pre-requisites

To contribute to the documentation, you will need to set your development environment.

You can create a virtual environment or conda environment and install the packages in
`docs/requirements.txt`.

```bash
# Create a virtual environment
python -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install the requirements
pip install -r docs/requirements.txt
```

or with conda

```bash
# Create a conda environment
conda create -n "grain-docs" python=3.12
# Activate the conda environment
conda activate grain-docs
# Install the requirements
python -m pip install -r docs/requirements.txt
```

## Building the documentation locally

To build the documentation locally, you can run the following command:

```bash
# From the root of the repository
sphinx-build -b html docs/source docs/_build/html
```

You can then open the generated HTML files in your browser by opening
`docs/_build/html/index.html`.
