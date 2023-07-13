# Machine Learning based Feature Engineering for Thermoelectric Materials by Design #

## Introduction ##

Availability of materials datasets through high performance computing has enabled the use of machine learning to not
only discover correlations and employ materials informatics to perform screening, but also take the first steps towards
materials by design. Computational materials databases are well-labelled and provide a fertile ground for predicting both
ground-state and functional properties of materials. However, a clear design approach that allows to predict materials
with desired functional performance does not yet exist. In this work, we train various machine learning models on a
dataset curated from a combination of materials project as well as computationally calculated thermoelectric electronic
power factor using constant relaxation time Boltzmann transport equation (BoltzTrap). We show that simple random
forest-based machine learning models outperform more complex neural network-based approaches on the moderately
sized dataset and also allow for interpretability. In addition, when trained on only cubic material systems, the best
performing machine learning model employs a perturbative scanning approach to find new candidates in materials project
that it has never seen before and automatically converges upon half-heusler alloys as promising thermoelectric materials.
We validate this prediction by performing Density Functional Theory and Boltztrap calculations to reveal accurate
matching. One of those predicted to be a good material, NbFeSb, has been studied recently by the thermoelectric
community; from this study, we propose four new half-Heusler compounds as promising thermoelectric materials –
TiGePt, ZrInAu, ZrSiPd and ZrSiPt. Our approach is generalizable to extrapolate into previously unexplored material spaces
and establishes an automated pipeline for high-throughput functional materials development.

## Overview of Project ##
![Image](https://github.com/Vaitesswar/Machine_Learning_for_Thermoelectric_Materials/assets/81757215/551216fe-00f2-414c-b5ac-1de820f242cf)
Description: (A) 4 machine learning models: Crystal Graph Convolutional Neural Network (CGCNN), Deep Neural Network (DNN), Random Forest (RF) and 
XG Boost (XGB) were trained independently for predicting power factor of thermoelectric materials. These models differ in terms of their architecture
and inputs as shown. (B) Random forest model, being the most accurate among the 4, was used to discover numerous new materials with good thermoelectric 
property (e.g. TiGePt, ZrInAu, NbFeSb, ZrSiPd and ZrSiPt) outside the training set.

## Prerequisites ##
The code is built with the following libraries:

- Python 3.6
- Anaconda
- PyTorch 1.3

## Data Preparation ##
The  json skeleton data for NW+UCLA dataset was obtained from the following link and was provided in CTR-GCN's Github page.
https://github.com/Uason-Chen/CTR-GCN

The skeleton data for NTU RGB+D and NTU RGB+D 120 datasets was obtained by extracting the 3D coordinates from raw .skeleton files obtained
from NTU ROSE lab.
https://rose1.ntu.edu.sg/dataset/actionRecognition/

Process the data using the appropriate Python files in data_gen folder for the different datasets. Note that the motion and bone data is preprocessed for NTU datasets. On the other hand, bone and motion data are processed during runtime for NW-UCLA dataset as seen in feeder_ucla.py in feeders folder.

## Models ##
The GCN models can be found under "models" folder for each dataset.

## Training ##
Prior to training, change lines 25 and 27 of training.py file in training folder to the appropriate model architecture and feeder respectively. Note that the feeder file varies for NTU and NW-UCLA datasets. The parameters of optimizer, training schedule and path to input data can be changed in main.py file. "processor" class will take in the parameters and will start training the model upon initiating "processor.start" command in main.py file.

## Testing ##
In main.py, update the "phase" attribute from "train" to "test" and specify the path to weights of the pretrained model in "weights" attribute.

## Pretrained weights and Test Results ##
The individual model weights and the predicted action labels for the test samples can be found in the "model weights and results" folder.

## Citing paper ##
Please refer to the following link for the full details of this model.
- https://dr.ntu.edu.sg/handle/10356/156866

If you find the repository or paper useful, please cite as follows.
- U S Vaitesswar (2022). Skeleton-based human action recognition with graph neural networks. Master's thesis, Nanyang Technological University, Singapore. https://hdl.handle.net/10356/156866
