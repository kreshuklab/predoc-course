# EIPP Theory@EMBL 2023 <!-- omit in toc -->

_Course material in previous years/events are managed by other branches._

You will design an image segemntation pipeline for immunofluorescence images of COVID-infected cells published in Microscopy-based assay for semi-quantitative detection of SARS-CoV-2 specific antibodies in human sera[[1]](https://www.biorxiv.org/content/10.1101/2020.06.15.152587v3). Please read the [Introduction](#introduction) before signing up to this challenge.

## Table of Contents <!-- omit in toc -->

- [Introduction](#introduction)
- [Challenge: Cell segmentation](#challenge-cell-segmentation)
  - [Nuclei segmentation](#nuclei-segmentation)
  - [Cell boundary segmentation](#cell-boundary-segmentation)
  - [Segmentation with seeded watershed](#segmentation-with-seeded-watershed)
  - [Segmentation results evaluation](#segmentation-results-evaluation)

## Introduction

- **Lecture**: You will learn the basic concepts of supervised machine learning and its application to the problem of image analysis. You will learn the difference between feature-based and deep machine learning, how to avoid some common pitfalls and where to begin to apply it to your own data.
- **Challenge**: You will design an image analysis pipeline for immunofluorescence images of COVID-infected cells (https://www.biorxiv.org/content/10.1101/2020.06.15.152587v2). In the first part of the challenge, you will explore algorithms to segment the cell nuclei. Then, you will investigate how segmented nuclei can help in the more challenging task of segmenting the entire cell membrane. In this challenge, you will learn how to use and adapt state-of-the-art bioimage analysis algorithms and combine them in a custom pipeline to quantify visual information from microscopy images.
- **Pre-requisites**: you will ace this challenge if you either have some prior experience with images (used Fiji before, for example) or some prior experience with coding in Python. If you don’t have either, try to get on a team where at least one member has it and you’ll learn along the way.

## Challenge: Cell segmentation

You will explore algorithms to segment individual cells in the IF images from the above
study as shown in the picture below:

![cell_segm](img/cell_segm.png?raw=true "Serum cells segmentation pipeline")

The input to the pipeline is an image consiting of 3 channels: the 'nuclei channel' (containing DAPI stained nuclei), the 'serum channel' (dsRNA antibody staining) and the 'infection channel' (ignored in this challenge).
The output from the pipeline is a segmentation image where each individual cell is assigned a unique label/number.
You can download the Covid assay dataset from [here](https://oc.embl.de/index.php/s/gfpnDykYgcxoM7y).
The dataset consist of 6 files containing the raw data together with ground-truth labels.
The data is saved using the HDF5 file format. Each HDF5 file contains two internal datasets:

- `raw` - containing the 3 channel input image; dataset shape: `(3, 1024, 1024)`: 1st channel - serum, 2nd channel - infection (**ignored**), 3 - nuclei
- `cells` - containing the ground truth cell segmentation `(1024, 1024)`
- `infected` - containing the ground truth for cell infection (at the nuclei level); contains 3 labels: `0 - background`, `1 - infected cell/nuclei`, `2 - non-infected cell/nuclei`

We recommend [ilastik4ij ImageJ/Fiji](https://github.com/ilastik/ilastik4ij) or [napari](https://napari.org/) for loading and exploring the data.

The actual segmentation task can be split in three parts:

1. Segmentation of the nuclei using the **nuclei channel**
2. Predicting cell boundaries using the **serum channel**
3. Segmentation of individual cells with a seeded watershed algorithm, given the segmented nuclei and the boundary mask

After successfully executing the 3 pipeline steps, you can qualitatively compare the segmentation results to the ground truth images (`cells` dataset).
For quantitative comparison one may use one of the common instance segmentation metrics, e.g. [Adapted Rand Error](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.adapted_rand_error).

More detailed description of the 3 steps can be found below.

### Nuclei segmentation

Explore algorithms for instance segmentation of the nuclei from the 'nuclei channel'.
After successfully segmenting the nuclei from the covid assay dataset, save the results in the appropriate format (tiff of hdf5),
since you'll need it in step 3.

### Cell boundary segmentation

In order to simplify the task of cell boundary prediction you will also use a pre-trained CNN.
This time we encourage you to use the [ilatik Neural Network Classification Workflow](https://www.ilastik.org/documentation/nn/nn).
Please download and install the **latest beta version** of ilastik in order to use the Neural Network Classification workflow (see: https://www.ilastik.org/download.html).

Then:

- open ilastik and create the `Neural Network Classification (Local)` project
- load a sample H5 image: `Raw Data -> Add New -> Add separate image -> (choose h5 file)`
  make sure to load only the serum channel (you need to extract the serum channel and save it in a separate h5 file beforehand). The size of the input should be `(1, 1024, 1024)`; **do not skip the singleton dimension**
- go to [BioimageIO](https://bioimage.io/), find [`CovidIFCellSegmentationBoundaryModel`](https://bioimage.io/#/?id=10.5281%2Fzenodo.5847355) and download ilastik weights by clicking on the ilastik `icon` and then `Download (Pytorch State Dict)`
- go to `NN Prediction` and click `Load model`; load the model file downloaded in the previous step
- after the model has been loaded successfully, click `Live Predict`; after the prediction is finished you can see the two output channels
  predicted by the network (i.e. foreground channel and cell boundaries channel) by switching between the layers in `Group Visibility` section (bottom left);
  you should see something like the image below:
  ![cell_segm](img/ilastik_nn_workflow.png?raw=true "NN workflow")
- go to `Export Data` and save the output from the network in hdf5 format for further processing

**Important note**
The network predicts 2 channels: the 1st channel contains a foreground(cells)/background prediction and the 2nd channel
contains the cell boundaries. You will need both channels for step 3.

### Segmentation with seeded watershed

Given the nuclei segmentation (step 1), the foreground mask and the the boundary prediction maps (step 2),
use the seeded watershed algorithm from the `skimage` (see [documentation](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed))
library in order to segment the cells in the serum channel.

Tip:
The watershed function is defined as follows:

```python
skimage.segmentation.watershed(image, markers=None, mask=None)
```

use boundary probability maps as `image` argument, nuclei segmentation as `markers` argument, and the foreground mask as the `mask` argument.

[environment.yml](environment.yml) can be used to create a conda environment with the necessary dependencies:

```bash
conda env create -f environment.yml
```

### Segmentation results evaluation

Compare your cell segmentation results with the ground truth (saved as `cells` dataset in the HDF5 files), using one/all of the metrics described below:

- [Adapted Rand Error](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.adapted_rand_error)
- [Variation of Information](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.variation_of_information)
- [Average Precision](https://github.com/hci-unihd/plant-seg/blob/master/evaluation/ap.py#L131)
