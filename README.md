# nanover-extensions-fieldlines

A NanoVer extensiosn for viewing electric field lines in VR.

## Setup the Environment

This package is installed with [uv](https://docs.astral.sh/uv/). We manage our environments with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)/[mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
To install this repository  use the following steps:

**Step 1**: Create a conda environment

```
mamba env create -n nanover-extensions-fieldlines python=3.13
```

**Step 2**: Activate your new environment

```
mamba activate nanover-extensions-fieldlines python=3.13
```

**Step 3**: Install the required dependencies

```
# nanover-extensions is a extension of nanover, so we naturally need it to be installed. This is currently only possible with conda
mamba install -c conda-forge -c irl nanover-server
# we use click for our cli interfaces, so that should be installed as well
mamba install click
```

**Step 4**: Install nanover-extensions-fieldlines.
```
pip install .

# Or optionally install in edit mode for changes made to this repo to be reflected in real time.
pip install -e .
```


## Github Repo Setup

To add your new project to its Github repository, firstly make sure you have created a project named **nanover-extensions-fieldlines** on Github.
Follow these steps to push your new project.

```
git remote add origin git@github.com:BristolChemVR/nanover-extensions-fieldlines.git
git branch -M main
git push -u origin main
```

## Built-in CLI Commands

This repository exposes a CLI scipt: `next-fieldlines`.
It can be run by simply typing `next-fieldlines` into the command line.

## PyPI Deployment

1. Register your project and create an API Token on [PyPI](https://pypi.org/).
2. Add the API Token to your projects secrets with the name `PYPI_TOKEN`
3. Create a new release on Github.
4. Create a new tag in the form `*.*.*`.

## Docs Generation + Publishing

Doc generation relies on [taskipy](https://github.com/taskipy/taskipy)
Doc generation is setup to scan everything inside `/src`, files with a prefix `_` will be ignored. Basic doc functions for generating, serving, and publishing can be done through these CLI commands:

```
# generate docs & serve
task docs

# serve docs
task serve

# generate static HTML docs (outputs to ./site/)
task html

# publish docs to Github Pages
task publish
```

Note: Your repo must be public or have an upgraded account to deploy docs to Github Pages.

## Dependabot Setup

1. Go to the "Settings -> Advanced Security" tab in your repository.
2. Under the "Dependabot" section enable the options you want to monitor, we recommend the "Dependabot security updates" at the minimum.

Dependabot is configured to do _weekly_ scans of your dependencies, and pull requests will be prefixed with "DBOT". These settings can be adjusted in the `./.github/dependabot.yml` file.

## References

- [Pattern](https://github.com/wyattferguson/pattern) - A modern cookiecutter template for your next Python project.

## License

MIT

## Contact

Maintained by [BristolChemVR](https://github.com/BristolChemVR)
