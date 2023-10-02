import argparse
import shutil
import sys
import time
import random
from random import sample
import csv
import functools
import json
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from pymatgen.core.structure import Structure

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        # elem_embedding is a dictonary containing feature vectors for the first 100 elements.
        # The keys (atom number according to periodic table) in 'string' format and the values (atom features) are in list format.
        # The keys are converted from string to integer format and restored as a dictionary.
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        # Removing repeated keys in the dictionary
        atom_types = set(elem_embedding.keys())
        
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        # Converting the format of values (atomic features) from list to numpy array and 
        # restored into the dictionary again.
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)
            
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor,dim = 0)
        self.std = torch.std(tensor,dim = 0)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
