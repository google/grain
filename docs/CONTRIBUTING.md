# Contributing to Grain



## Contributing to the Grain project documentation

### Pre-requisites

To contribute to the documentation, you will need to set your development
environment.

You can create a virtual environment or conda environment and install the
packages in `docs/requirements.txt`.

```bash
# Create a virtual environment
python3 -m venv .venv
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
python3 -m pip install -r docs/requirements.txt
```

### Building the documentation locally

To build the documentation locally, you can run the following command:

```bash
# Change to the docs/ directory
cd docs
sphinx-build -b html . _build/html
```

You can then open the generated HTML files in your browser by opening
`docs/_build/html/index.html`.

## Documentation via Jupyter notebooks

The `pygrain` documentation includes Jupyter notebooks that are rendered
directly into the website via the [myst-nb](https://myst-nb.readthedocs.io/)
extension. To ease review and diff of notebooks, we keep markdown versions of
the content synced via [jupytext](https://jupytext.readthedocs.io/).

Note you will need to install `jupytext` to sync the notebooks with markdown
files:

```bash
# With pip
python3 -m pip install jupytext

# With conda
conda install -c conda-forge jupytext
```

### Adding a new notebook

We aim to have one notebook per topic or tutorial covered. To add a new notebook
to the repository, first move the notebook into the appropriate location in the
`docs` directory:

```bash
mv ~/new-tutorial.ipynb docs/tutorials/new_tutorial.ipynb
```

Next, we use `jupytext` to mark the notebook for syncing with Markdown:

```bash
jupytext --set-formats ipynb,md:myst docs/tutorials/new_tutorial.ipynb
```

Finally, we can sync the notebook and markdown source:

```bash
jupytext --sync docs/tutorials/new_tutorial.ipynb
```

To ensure that the new notebook is rendered as part of the site, be sure to add
references to a `toctree` declaration somewhere in the source tree, for example
in `docs/index.md`. You will also need to add references in `docs/conf.py`
to specify whether the notebook should be executed, and to specify which file
sphinx should use when generating the site.

### Editing an existing notebook

When editing the text of an existing notebook, it is recommended to edit the
markdown file only, and then automatically sync using `jupytext` via the
`pre-commit` framework, which we use to check in GitHub CI that notebooks are
properly synced.
For example, say you have edited `docs/tutorials/new_tutorial.md`, then
you can do the following:

```bash
pip install pre-commit
git add docs/tutorials/new_tutorial.*          # stage the new changes
pre-commit run                       # run pre-commit checks on added files
git add docs/tutorials/new_tutorial.*          # stage the files updated by pre-commit
git commit -m "Update new tutorial"  # commit to the branch
```

### Release procedure

Grain release can be done by running the "Build and Publish Release" workflow.
It builds wheels for all supported platforms and uploads them to PyPI.

IMPORTANT! Please remember to bump `version` entry in `MODULE.bazel` and
`pyproject.toml` files after each release. This will allow the nightly workflow
to have correct identifier. E.g. if you just released `0.2.10`, push a commit
that moves version entry to the next anticipated release, such as `0.2.11` or
`0.3.0`.
