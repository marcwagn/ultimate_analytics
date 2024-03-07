# ultimate_pipeline

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.3`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
python -m pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

# Development

## Dev machine setup
### Supervisely
Create `.env` file (or`~/supervisely.env`) with Supervisely API key:
```
SERVER_ADDRESS="https://app.supervise.ly"
API_TOKEN="<secret_api_key>
```
### Installing development/build dependencies

## Managing project dependencies

TL;DR: Declare **direct** dependencies in `requirements.txt` for `pip` installation. Then use [pip-tools](https://github.com/jazzband/pip-tools) to generate the lock file with indirect depencies.

### Adding a new runtime dependency
1. Install the dependency with pip/pipx
2. If this is a direct project runtime dependency (and its modules are being imported in some project files), add the package requirement along with the version specification to `requirements.txt`
e.g. `supervisely==6.73.42`
3. Run `pip-compile requirements.txt -o requirements.lock` to generate full list of project dependencies, including transitive dependencies.
4. Commit both `requirements.txt` and `requirements.lock` to Git.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

### Adding a new dev dependency
To add a new development/build dependency, edit `project.optional-dependencies` section in `pyproject.toml`file.

### Installing project runtime dependencies
1. (/Only in dev environment/) Activate your Python environment (e.g. with `conda activate <envname>`)
2. Run `pip install -r requirements.lock`

### Troubleshooting dependencies

You can use `pipdeptree` to visualise pip dependencies:
```
pip install pipdeptree
pipdeptree
```


## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
