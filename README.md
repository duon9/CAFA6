# CAFA6 Protein Function Prediction

This project is a machine learning solution for the CAFA6 protein function prediction challenge. It utilizes various techniques, including sequence alignment, protein language model embeddings, and deep learning models to predict protein functions.

## Project Structure

The project is organized into the following directories:

- **`diamond/`**: Contains a Jupyter notebook (`diamond.ipynb`) for running the DIAMOND tool for fast protein sequence alignment against a database. This is likely used for homology-based function annotation.

- **`embedding/`**: This directory contains scripts to generate protein embeddings using different pre-trained protein language models:
    - `ankh3XL_5_7B.py`: Generates embeddings using the Ankh model.
    - `esm2_3b.py`: Generates embeddings using the ESM-2 model.
    - `prott5_3b.py`: Generates embeddings using the ProtT5 model.

- **`gradientboosting/`**: This section of the project uses a gradient boosting model for prediction.
    - `train.py`: The main script to train the gradient boosting model.
    - `config.py`: Configuration file for the training process.
    - `pyboost/`: Contains a custom implementation of a gradient boosting library.

- **`mlp/`**: This directory contains the implementation of a Multi-Layer Perceptron (MLP) model for the prediction task.
    - `train.py`: The main script to train the MLP model.
    - `config.py`: Configuration for the MLP model and training.
    - `losses.py`: Custom loss functions.
    - `metrics.py`: Evaluation metrics.
    - `protein_dataset.py`: A PyTorch Dataset class to handle the protein data.
    - `protein_module.py`: The PyTorch Lightning module defining the MLP architecture and training logic.

- `requirements.txt`: A list of Python dependencies required to run the project.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install PyBoost:**
    Navigate to the `gradientboosting/pyboost` directory and install the custom library.
    ```bash
    cd gradientboosting/pyboost
    pip install .
    cd ../..
    ```

## Usage

The project consists of several independent components that can be run in sequence.

### 1. Sequence Alignment (DIAMOND)

The `diamond/diamond.ipynb` notebook provides instructions and code to perform sequence alignment. You will need to have the DIAMOND tool installed and a protein database available.

### 2. Generating Embeddings

To generate embeddings for your protein sequences, you can use the scripts in the `embedding/` directory. For example, to use the ProtT5 model:

```bash
python embedding/prott5_3b.py --input your_sequences.fasta --output embeddings.h5
```

Make sure to check the arguments for each script to see the available options.

### 3. Training the Models

Both the gradient boosting and MLP models have a `train.py` script.

**Gradient Boosting:**

Navigate to the `gradientboosting` directory and run the training script:

```bash
cd gradientboosting
python train.py
```

You can modify the `config.py` file to change the training parameters, input data paths, and other settings.

**MLP:**

Navigate to the `mlp` directory and run the training script:

```bash
cd mlp
python train.py
```

Similarly, you can adjust the MLP training process by editing the `config.py` file in the `mlp` directory.

## Models

### Gradient Boosting

The `gradientboosting` directory contains a custom-built gradient boosting library (`pyboost`). This model is trained on the generated embeddings and/or other features to predict protein function.

### Multi-Layer Perceptron (MLP)

The `mlp` directory contains a deep learning model implemented in PyTorch and PyTorch Lightning. The `protein_module.py` file defines the architecture of the MLP, and the `train.py` script handles the training and evaluation. The model is designed to take protein embeddings as input.
