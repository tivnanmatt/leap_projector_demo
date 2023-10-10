# LEAP Projector Demo

Minimal Working Example of 3D CT Reconstruction with the LEAP Projectors


## About
This project provides a Python script for CT volume reconstruction using forward and back projection methods. By utilizing the leaptorch package for volume projections and the torch package for tensor computations, this code offers both ART (Algebraic Reconstruction Technique) and FBP (Filtered Back-Projection) based reconstructions. The provided code also showcases the use of the ramp filter in the reconstruction process.

## Installation

1. Create and activate the conda environment:

```bash
$ conda env create --file environment.yml --name myEnv
$ conda activate myEnv
```

2. Install PyTorch and related libraries:

```bash
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Clone and install the LEAP library:

```bash
$ git clone https://github.com/LLNL/LEAP.git
$ cd LEAP
$ pip install .
```

## Usage

Once the required packages are installed and the environment is set up, you can run the main Python script. This script performs the following:

* Initializes parameters for the imaging setup such as volume dimensions, voxel dimensions, number of angles, rows, columns, and other specifics.
* Defines the type of beam used (cone beam vs parallel beam).
* Loads a sample volume data TCGA_LIHC_000401.npy from the data directory.
* Computes the ground truth projections using the Projector class from leaptorch.
* Displays and saves the true volume and projections.
* Provides utility functions for forward and back projection operations.
* Implements Preconditioned Gradient Descent using the Ramp Filter as an approximation to the Hessian.

