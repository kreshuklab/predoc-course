# predoc-course-2020

You will design an image analysis pipeline for immunofluorescence images of COVID-infected cells published in [Microscopy-based assay for semi-quantitative detection of SARS-CoV-2 specific antibodies in human sera](https://www.biorxiv.org/content/10.1101/2020.06.15.152587v2). 
In this challenge, you will learn how to use and adapt state-of-the-art bioimage analysis algorithms and combine them in 
a custom pipeline to quantify visual information from microscopy images.

## Cell segmentation
In the first part of this challenge you will explore algorithms to segment individual cells in the IF images from the above 
study as shown in the picture below:

![cell_segm](img/cell_segm.png?raw=true "Serum cells segmentation pipeline")

The input to the pipeline is an image consiting of 3 channels: the 'nuclei channel' (containing DAPI stained nuclei), the 'serum channel' (dsRNA antibody staining) and the 'infection channel' (ignored in this challenge).
The output from the pipeline is a segmentation image where each individual cell is assigned a unique label/number.
You can download the Covid assay dataset from [here](https://oc.embl.de/index.php/s/gfpnDykYgcxoM7y).
The dataset consist of 6 files containing the raw data together with ground-truth labels.
The data is saved using the HDF5 file format. Each HDF5 file contains two internal datasets:
* `raw` - containing the 3 channel input image; dataset shape: `(3, 1024, 1024)`: 1st channel - serum, 2nd channel - infection (**ignored**), 3 - nuclei
* `cells` - containing the ground truth cell segmentation `(1024, 1024)`
* `infected` - containing the ground truth for cell infection (at the nuclei level); contains 3 labels: `0 - background`, `1 - infected cell/nuclei`, `2 - non-infected cell/nuclei` 

We recommend [ilastik4ij ImageJ/Fiji](https://github.com/ilastik/ilastik4ij) for loading and exploring the data. 

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

### Cell boundary segmentation using pre-trained model and ilastik's Neural Network workflow
In order to simplify the task of cell boundary prediction you will also use a pre-trained CNN.
This time we encourage you to use the [ilatik Neural Network Classification Workflow](https://www.ilastik.org/documentation/nn/nn).
Please download and install the latest beta version of ilastik in order to use the Neural Network Classification workflow (see: https://www.ilastik.org/download.html).

After downloading and installing ilastik in your system, please follow the [instructions](https://github.com/ilastik/tiktorch)
required to run the Neural Network workflow with ilastik.

Given the successful setup of the Neural Network workflow, please download the pre-trained 2D U-Net model trained to predict
cell boundaries from the serum channel from [here](https://oc.embl.de/index.php/s/wCw0u5dJ5J3SDOE).
Then:
* start tiktorch server: see `https://github.com/ilastik/tiktorch`
* open ilastik and create the `Neural Network Classification (Beta)` project
* load a sample H5 image: `Raw Data -> Add New -> Add separate image -> (choose h5 file)`
make sure to load only the serum channel; you need to extract the serum channel and save it in a separate h5 file, the size should be `(1, 1024, 1024)`; **do not skip the singleton dimension**
* go to `Server Configuration` and click `Get Devices` (default settings for the host and port should be correct);
you should see at least the `cpu` device in the list; if you have a `cuda` capable device on your system then select it; click `Save`
* go to `NN Training`, click `Load model` and select the `UNet2DSarsCoV2SerumBoundary.zip` file you downloaded from [here](https://oc.embl.de/index.php/s/wCw0u5dJ5J3SDOE)
* after the model has been loaded successfully, click `Live Predict`; after the prediction is finished you can see the two output channels
predicted by the network (i.e. foreground channel and cell boundaries channel) by switching between the layers in `Group Visibility` section (bottom left);
you should see something like the image below:
![cell_segm](img/ilastik_nn_workflow.png?raw=true "NN workflow")
* go to `Export Data` and save the output from the network in hdf5 format for further processing

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
 
## Nuclei classification and shape characterization
In this part of the challenge you will train a classifier to distinguish
between infected/non-infected cells and characterize the shapes of the
(segmented) nuclei.

### Nuclei classification
**Task**
(Implement and) train a classifier that is able to group cell into the categories infected/non-infected.
To see how your model performs, reserve the nuclei from one of the 6 images for testing.

You are completely free in the way you solve this task, i.e. which model you use, if you use one from
a library or implement it yourself etc.

**General hints**
- [sklearn](https://scikit-learn.org/stable/) is a great library for classical machine learning in python and contains classifiers such as random forst, SVM etc.
- [pytorch](https://pytorch.org/) is a great library if you want to use a neural network
- [numpy](https://numpy.org/) is fundamental package for scientific computing which you will definetly need for this task

All these packages (and more) can be installed via [conda] (https://www.anaconda.com/).

In the following we will give you some hints on how to solve this task with a neural network.
The packages needed for that are contained in the `environment.yml` and can be installed via `conda env create -f environment.yaml`

#### Data handling
First, you should transform your data in a way that you can locate each nucleus and give it to the network for classification.
This can be done, e.g., by generating a sequence of bounding boxes for each image in which each bounding box contains the
location of one nucleus.

You can open the `.h5` files by using pythons `with` statement and the `h5py.File` function. From there you can extract
all the information you need (have a look at [h5py doc](https://docs.h5py.org/en/stable/index.html) if you are stuck)
to generate nuclei bounding boxes.

Next, you have to implement a custom `Dataset` class (see [pytorch Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)), which you will later use to sample nuclei for training your network.
The class could look like this:

```class CovidDataset(Dataset):
    def __init__(self, raw_images, nuclei_segmentations, infected_masks):
        self._raw = raw_images
        self._nuclei_segmentations = nuclei_segmentations
        self._infected_masks = infected_masks
        self.nuclei_bounding_boxes = self.compute_bounding_boxes_from_nuclei_segmentation()
        
    def compute_bounding_boxes_from_nuclei_segmentation(self):
        TODO
        
    def crop_out_nuclei_from_raw_image(self, bounding_box):
        TODO
        
    def get_label_from_ground_truth_mask(self, bounding_box):
        TODO

    def __getitem__(self, idx):
        bounding_box = self.nuclei_bounding_boxes[idx]
        raw_nuclei = self.crop_out_nuclei_from_raw_image(bounding_box)
        label = get_label_from_ground_truth_mask(bounding_box)
        return (raw_nuclei, label)

    def __len__(self):
        TODO
```

The last step here is to create a `DataLoader` object which will enable you to draw random samples from your dataset class
(have a look at [pytorch DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader)).

#### Model
Next, you have to implement a neural network. Have a look at [pytorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) how to do that. It should have the following form

```import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # network layers

    def forward(self, nuclei_images):
        # specifics of the forward computation
```

pytorch's `nn` module contains many different layers and it is easy to get lost. Focus on the `nn.Conv2d`, `nn.MaxPool2d` and
`nn.Linear` layers to build your network. You'll also need an activation function; have a look at `torch.nn.functional.relu`.

#### Loss function and optimizer
To make your network learn, you have to give it feedback on how well it is performing. This is done using a loss function.
You can use the `torch.nn.BCELoss` for that.

After having computed the loss, you want to propagate the feedback signal back through the network to update its parameters.
The `torch.optim` package takes care of this. It contains many optimizers which follow the same principle but differ in
the way they exactly update the weights. Usually `torch.optim.Adam` is a good choice but you can also use the classical
`torch.optim.SGD` optimizer.

#### Training loop
The training loop contains all steps to train a network. More specifically, it starts with drawing samples from the data loader, which are then fed to the network to compute the forward pass. The networks outputs the predictions, which is the
label infected/not-infected in our case, which are then used in combination with the ground truth labels to compute a loss.
After having scored the predictons, we back propagate the error through the network and update our weights using the optimizer. A simple training routine can look like this

```for epoch in range(epochs):
    for sampled_nuclei, lables in data_loader:
        # zero parameter gradients
        optimizer.zero_grad()
        
        # forward
        predictions = network(sampled_nuclei)
        
        # loss + backward
        loss = criterion(predictions, labels)
        loss.backward()
        
        # optimizer update
        optimizer.step()
```

**Note**
If you are stuck, have a look at the various pytorch tutorial, e.g. [cifar classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### Charactersiation of the nuclei shapes
**Task**
Find out what shape descriptors are. Which ones are commonly used in cell biology?
Select a subset of the ones that you find interesting and represent your data with them. Finally, create a umap/pca plot of the nuclei population using shape descriptors.
