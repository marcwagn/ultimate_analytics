# ultimate_pipeline

## Overview

This is the data processing pipeline for the Ultimate project.

## How to run the Kedro pipeline

You can run the Kedro project with:

```
kedro run
```

By default, it downloads Ultimate project images and annotations from Supervisely and converts them to format used by YOLO.

To run the pipeline only from a specific step, e.g. to skip Supervisely download:
```bash
kedro run --from-nodes=convert_supervisely_annotations_to_yolo_detect_dataframe_node
```

## Data folder structure

Training data is not in source control and needs to be downloaded from Supervisely by `kedro run`.
After a successful pipeline run, the `data` folder will have following subfolders:
- `raw` - images and annotations downloaded from Supervisely
- `processed` - images and annotations ready for Machine Learning training with YOLO object detection 

For Kedro datasets description, see `conf/base/catalog.yml`.

For YOLO training dataset description, see `conf/base/ultimate_detect.yml`.

## How to test the Kedro project

Unit tests can be run as follows:

```bash
python -m pytest
```

Code coverage threshold can be configured in `pyproject.toml` file under the `[tool.coverage.report]` section.

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
```bash
pip install pipdeptree
pipdeptree
```

# Machine Learning training
Machine Learning training requires the training data being downloaded and processed by Kedro first, but it's performed outside of the Kedro pipeline.


Object detection training is performed with [Ultralytics YOLO](https://docs.ultralytics.com/quickstart/).

## Training
Example:
```bash
 yolo detect train data=conf/base/ultimate_detect.yml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=3840 batch=2
```

### Resume training
Resuming an interrupted training
```bash
 yolo detect train resume data=conf/base/ultimate_detect.yml model=runs/detect/train10/weights/last.pt
```
where `runs/detect/train10` are the results of the previous YOLO training run.

## Validation
```bash
yolo detect val model=runs/detect/train10/weights/best.pt
```
where `runs/detect/train10` are the results of the previous YOLO training run.


## Prediction
```bash
yolo detect track model=runs/detect/train10/weights/best.pt save=True save_txt=True save_conf=True agnostic_nms=True source=<some_ultimate_frisbee_match.mp4>
```
/Note/: Prediction is performed by the main web app of the Ultimate project.

# Notebooks

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
