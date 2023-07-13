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

## Methodology ##
![Image 2](https://github.com/Vaitesswar/Machine_Learning_for_Thermoelectric_Materials/assets/81757215/22bb0e7a-103c-446d-bd65-453e3ee8f497)
The 3 phases involved in the forward model training are retrieving data from open-source material databases, processing the retrieved data to obtain the necessary features and developing optimized forward machine learning models based on these features 

## Prerequisites ##
The code is built with the following libraries:

- Python 3.6
- Anaconda
- PyTorch 1.3
- Scikit-learn
- Pymatgen

## Data Preparation ##
In this project, the dataset was obtained from the work of _Ricci, F. et al. An Ab Initio Electronic Transport Database for
Inorganic Materials. Sci. Data 4, 170085 (2017)_ who published an electronic transport database for inorganic materials. This dataset was developed by retrieving the electronic band structures from Materials Project and utilizing them to compute the thermoelectric properties of materials using a BTE package called BoltzTrap 27 . This dataset contains more than 23,000 entries of multi-level data for 8059 materials and is stored in separate json files. Particularly, there would be multiple entries for each material, each with a different temperature, doping level and carrier type. These 23,000 json files were flattened and compiled into a single csv file for ease of use for ML application. The flattened dataset was augmented with atomic properties data, retrieved from Materials Project Database (MPD) 21 using the Matminer 19,20 Python package. In short, CGCNN has a total of 15 input features while DNN, XG Boost and RF models have a total of 26 input features. The data can be found in "Data" folder.

## Models ##
The different machine learning models and inverse design algorithmic model can be found under "Models" folder.

## Training & Testing ##
Run the respective machine learning .py files to train and test the models in Jupyter Notebook using "Restart & Run All" command. Ensure that the paths to the respective external files are set correctly.
 
## References ##
Crystal graph convolutional neural networks. Available at: https://github.com/txie-93/cgcnn.
