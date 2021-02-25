# Cassava-Leaf-Disease-Classification

## Introduction
This is a solution for [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) by [me](https://github.com/t0efL), [Maksim Zhdanov](https://github.com/xzcodes) and [Emin Tagiev](https://github.com/Emilien-mipt).  

In this repository you can find my pipeline and our solution for this competition. In the first part of README I'll describe our solution and in the second part I'll describe the repository's structure with the descriptions of all the files.

Source write-up is [here](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/220586).  
You can find our full report (33 pages) about this competition [here](https://docs.google.com/document/d/1TNTfrDrhYSAAgL_L6gIX5stw1Lfm76XJ-QY7yS4UfTc/edit?usp=sharing) (or [here](https://github.com/t0efL/Cassava-Leaf-Disease-Classification/blob/main/FullReport.docx)).

## Solution

### Knowledge distillation
We did a really soft knowledge distillation. We hadn't time to do a lot of experiments, so we had only one take to relable the dataset. We explored the forum and found out that there are at least 500 diseased images which are labeled as healthy. Other types of mistakes weren't frequently mentioned. So we decided to make knowledge distillation soft targets only for 500 samples. We trained 5 folds of SWSL ResNeXt 101 8D and predicted labels for all images from each validation part and for each label. Then we saved the predictions and confidences for each image and chose a threshold for confidence, which was used to find ~500 samples where extremely confident predictions didn't match ground truth labels. Then we trained Efficient Net B4 with obtained labels and got the CV boost from 0.899 to 0.915. Relabeled data was used in validation part as well.

### External data
##### 2019 data
We used [2019 data with removed duplicates](https://www.kaggle.com/tahsin/cassava-leaf-disease-merged) for training. We used it only for training and didn't put it in the validation part. We've also applied soft relabling for it.
##### Mendeley Leaves
We also used [Mendeley Leaves](https://www.kaggle.com/nroman/mendeley-leaves) for training. This dataset contains images of other plants. The dataset is not fully labeled. The first part of the all images are labeled just as healthy and the rest of them are labeled just as diseased. So the second part had to be pseudo-labeled to be able to use it for training. We used 5 folds of SWSL ResNeXt 101 8D for this task. We predicted labels for each diseased image and checked model confidence for each one. I used a 0.55 confidence threshold - the same  was calculated for relabeling  ~500 samples in the source train set. (It’s NOT ranged from 0 to 1, it has much more amplitude including negative values.). Finally, we get  2767 / 4334 samples, including the healthy ones. The external data images weren’t used in the validation and placed only in the training part.

### TTA
We used simple technique for our TTA: average of the default image and horizontally flipped image. There are two reasons of selecting such a small number of transforms for TTA: time limit - well-structured ensemble usually performs better than TTA,  double TTA is the most stable variant in this competition according to our experiments - averaged and weighted 4x or 16x TTA aren't that good to be considered for our solution, LB scores might differ with them in range [-0.007, +0.007]. 

### Augmetations

Successful:
- Horizontal Flip
- Vertical Flip
- Blur
- 360 rotates 
- RandomBrigtnessContrast 
- ShiftScaleRotate
- HueSaturationValue
- CutMix
- MixUp
- FMix 

25% probability for each of past three augmentations.

Unsuccessful:
- ElasticTransform
- Grid Distortion
- RandomSunFlare
- GaussNoise
- Coarse dropout
- Optical Distortion

During warm up stage only horizontal flip was used.

### Training
- optimizer - Ranger
- learning rate - 0.003
- epoch - 30
- warm up epochs - 3
- early stopping - 8
- Loss Function - Cross Entropy loss with label smoothing (0.2 smoothing before using knowledge distillation and 0.1 after it)
- Progressive image size, start_size=256, final_size=512, size_step=32. Size starts increasing after the warm up stage finishes.
- pretrained - True
- Frozen BatchNorm layers
- scheduler  - CosineBatchDecayScheduler with gradual warm up (custom impelementation)

Initially, we tried to impelement Cosine Decay Scheduler that steps every batch and has wide range of customization ways, but finally we created the custom scheduler we used here. Its code is [here](https://github.com/t0efL/Cassava-Leaf-Disease-Classification/blob/main/custom_functions/scheduler.py). Here is a plotted learning rate for this scheduler:

![](https://github.com/t0efL/Cassava-Leaf-Disease-Classification/blob/main/images/lr_plot.png)

Mixed precision training with gradient accumulation (iters_to_accumulate=8) ==> big boost on CV

### Models
- SWSL ResNeXt 101 8D
- SWSL ResNeXt 50
- Efficient Net B4
- Efficient Net B4 NS
- Inception V4
- DenseNet 161

Here are the table representing perfomance of these and some others models:
![](https://github.com/t0efL/Cassava-Leaf-Disease-Classification/blob/main/images/table.jpg)

### Final ensembles
We submitted two different ensemble approaches with the same 6 models (5 folds each one). 

1) Simple Averaging Ensemble
**2019 Private LB: 0.92712; 2020 Public LB: 0.898; 2020 Private LB: 0.896**

2) MaxProb Ensemble
Max Probability (or Max Confidence) ensemble allows us to choose the dominating model in the prediction process. In this case, if several models predict different labels, we take the prediction from a model with the biggest confidence.
**2019 Private LB: 0.93197; 2020 Public LB: 0.893; 2020 Private LB: 0.897**

Here is the scheme of our ensembles:
![](https://github.com/t0efL/Cassava-Leaf-Disease-Classification/blob/main/images/scheme.png)

## Repository structure
- **custom_functions** - folder containing all the custom functions and classes
  - scheduler.py - custom scheduler (CosineBatchDecayScheduler)
  - lossfn.py - custom loss function with label smoothing
  - optimizer.py - Ranger optimizer (copied from [here](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer))
  - augmentations.py - custom augmentations implementations (cutmix, mixup, fmix)
- **experiments** - bonus code for experiments
  - knowledge_distillation.py - code for getting new labels and model's confidence
  - kd_process.py - processing of new labels according model's condfidence
  - pseudolabeling.py - code for pseudo-labeling of external data
  - pl_process.py - processing of pseudo-labeled labels according model's confidence
- **images** - folder containing all the images for README
  - lr_plot.jpg - learning rate changes plot
  - table.jpg - table with the training results
  - scheme.jpg - scheme of our final ensemble
- **FullReport.docx** - our full report for this competition
- **config.py** - main config for training, you can set up everything there
- **data_functions.py** - module containing functions and classes for processing and preparing the data for training
- **fmix.py** - code for fmix function (copied from [here](https://github.com/ecs-vlc/FMix/blob/master/fmix.py))
- **main.py** - main module, run it to start the training
- **train.csv** - annotation file for training containing image names and labels for source dataset, 2019 dataset and pseudo-labeled part of the external data dataset
- **train_functions.py** - main functions for the training
- **utils.py** - module containing all the auxiliary functions
- **weights_transformer.py** - code for cropping weights (leaves only the state of the model in the file)

*Don't deal with the noise...*
