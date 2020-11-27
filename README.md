# predoc-course-2020

You will design an image analysis pipeline for immunofluorescence images of COVID-infected cells published in [Microscopy-based assay for semi-quantitative detection of SARS-CoV-2 specific antibodies in human sera](https://www.biorxiv.org/content/10.1101/2020.06.15.152587v2). 
In this challenge, you will learn how to use and adapt state-of-the-art bioimage analysis algorithms and combine them in 
a custom pipeline to quantify visual information from microscopy images.

## Cell segmentation
In the first part of this challenge you will explore algorithms to segment individual cells in the IF images from the above 
study as shown in the picture below:

![cell_segm](img/cell_segm.png?raw=true "Serum cells segmentation pipeline")

The input to the pipeline is an image consiting of 2 channels: the 'nuclei channel' (containing DAPI stained nuclei) and the 'serum channel' (dsRNA antibody staining). The output from the pipeline is a segmentation image where each individual cell is assigned a unique label/number.
You can download the images, together with ground-truth labels from [here](https://github.com/hci-unihd/antibodies-nuclei/tree/master/groundtruth) (TODO: upload to OwnCloud).
The covid assay dataset is saved using the HDF5 file format. Each HDF5 file contains two internal datasets:
* `raw` - containing the 2 channel input image; dataset shape: `(2, 1024, 1024)` 
* `cells` - containing the ground truth cell segmentation `(1024, 1024)`

We recommend [ilastik4ij ImageJ/Fiji](https://github.com/ilastik/ilastik4ij) for loading and exploring the data. 

The actual segmentation task can be split in three parts:
1. Segmentation of the nuclei using the **nuclei channel**
2. Predicting cell boundaries using the **serum channel**
3. Segmentation of individual cells with a seeded watershed algorithm, given the segmented nuclei and the boundary mask

After successfully executing the 3 pipeline steps, you can qualitatively compare the segmentation results to the ground truth images (`cells` dataset).
For quantitative comparison one may use one of the common instance segmentation metrics, e.g. [Adapted Rand Error](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.adapted_rand_error).

More detailed description of the 3 steps can be found below.

### Nuclei segmentation with pre-trained StarDist model
We recommend to use [StarDist](https://arxiv.org/abs/1806.03535) for nuclei segmentation. The easiest way is to use the
StarDist ImageJ/Fiji plugin, which contains an already pre-trained neural networks and can be used out of the box.
Please visit the [plugin website](https://imagej.net/StarDist) and follow the instruction to install the plugin.
In order to segment the nuclei from the nuclei channel please use the `Versatile (fluorescent nuclei)` model,
pre-trained on the `DSB 2018` challenge, which contains data similar to our covid assay dataset.
After successfully segmenting the nuclei from the covid assay dataset, save the results in the appropriate format (tiff of hdf5),
since you'll need it in step 3.

### Cell boundary segmentation using pre-trained model and ilastik's Neural Network workflow


### Segmentation with seeded watershed


## Analyze the distribution of shapes in the segmented cells/nuclei population
For Virginie and Johannes... 