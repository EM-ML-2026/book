# Start here

## Preliminary knowledge

To follow the course effectively, participants should be comfortable with basic Python programming. This includes working with variables, functions, loops, conditionals, classes, and modules. They should also be able to run Python code, modify example scripts, and interpret error messages.

In particular, participants should have experience with:

- **NumPy** for numerical arrays, vectorized operations, and basic linear algebra
- **Matplotlib** for plotting and visualization
- basic use of external Python libraries in computational workflows

Ideally, participants should also be familiar with:

- linear algebra
- calculus
- probability and statistics
- ordinary and partial differential equations

Some exercises use modern machine learning libraries and ecosystems such as
[scikit-learn](https://scikit-learn.org),
[PyTorch](https://pytorch.org),
[JAX](https://jax.readthedocs.io), and
[Equinox](https://docs.kidger.site/equinox/).
Prior experience with these libraries is useful, but not required.

## Running the code or notebooks

You are free to use the coding IDE or platform of your choice. Two practical options are running locally on your own machine or using Google Colab.

### Option 1: Google Colab

Google Colab provides a hosted Python environment that requires no local installation. It is convenient for a quick start, although execution may be slower than on your own machine.

Open [Colab](https://colab.research.google.com/), sign in with your Google account, and upload the python scripts or [jupyter-notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) you want to run.

Some notebooks download datasets automatically. Others require you to provide data manually. In those cases, upload the data to Google Drive and mount the drive from Colab:

```python
# Connect to Google Colab
from google.colab import drive

# This will prompt for authorization to access your Google Drive from Colab.
drive.mount('/content/drive', force_remount=True)

# After mounting, you can navigate to a specific folder using the usual UNIX cd command.
# Replace 'your_folder_path' with the actual path of your folder inside Google Drive.
folder_path = '/content/drive/MyDrive/cnn_application/'  # Example path

%cd "$folder_path"
```

### Option 2: Coding locally

Running locally is usually faster and gives you full control over the IDE and Python environment. A virtual environment is recommended.

#### Using `venv`

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

To deactivate the virtual environment:

```bash
deactivate
```

#### Using `conda`

```bash
conda env create -f environment.yml
conda activate ml-course
```

To deactivate the conda environment:

```bash
conda deactivate
```

## Repository files

This repository includes two environment configuration files:

- `requirements.txt` for a `pip`-based setup
- `environment.yml` for a `conda`-based setup

Use one of them, not both at the same time.
