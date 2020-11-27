# predoc-course-2020

You will design an image analysis pipeline for immunofluorescence images of COVID-infected cells published in [Microscopy-based assay for semi-quantitative detection of SARS-CoV-2 specific antibodies in human sera](https://www.biorxiv.org/content/10.1101/2020.06.15.152587v2). In this challenge, you will learn how to use and adapt state-of-the-art bioimage analysis algorithms and combine them in a custom pipeline to quantify visual information from microscopy images.


As part of this challeng you will explore algorithms to segment individual cells in the IF images from the above study as shown in the picture below:



The input to the pipeline is an image consiting of 2 channels: the 'nuclei channel' (containing DAPI stained nuclei) and the 'serum channel' (dsRNA antibody staining). The output from the pipeline is a segmentation image where each individual cell is assigned a unique label/number.
You can download the images, together with ground-truth labels from [here](https://github.com/hci-unihd/antibodies-nuclei/tree/master/groundtruth) (TODO: upload to OwnCloud).

The segmentation task can be split in three parts:
1. Segmentation of the nuclei using the **nuclei channel**
2. Predicting cell boundaries using the **serum channel**
3. Segmentation of individual cells with a seeded watershed algorithm, given the segmented nuclei and the boundary mask.

### Nuclei segmentation with ore-trained StarDist model


### Cell boundary segmentation using pre-trained model and ilastik's Neural Network workflow


### Segmentation with seeded watershed

