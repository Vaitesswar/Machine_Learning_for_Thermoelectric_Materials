
# coding: utf-8

# In[ ]:


from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module): # Inheriting the attributes from the Module class under pytorch.nn
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.
        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__() # This is necessary to state that attirbutes from the super class (nn.Module) are inherited by our new object.
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        # Concatenate 2 atom feature vectors and their bond feature vectors
        # y = Wx + b (Linear function) -> Eq.5
        # Input feature length = 2*self.atom_fea_len + self.nbr_fea_len
        # Output feature length = 2*self.atom_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len) 
        self.sigmoid = nn.Sigmoid() # Sigmoid function in Eq.5
        self.softplus1 = nn.Softplus() # non-linear function g in Eq.5
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len) 
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        Parameters
        ----------
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        Returns
        -------
        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution
        """
        # TODO will there be problems with the index zero padding?
        # N is the number of atoms in crystal and M is the number of neighbors for each atom.
        N, M = nbr_fea_idx.shape 
        # convolution
        # Note that atom feature vectors are stacked according to their numbering in the crystal lattice.
        # Hence, the following line for filtering the data based on nbr_fea_idx works. 
        # atom_nbr_fea is a 3 dimensional matrix (number of atoms (x) * number of neighbors (y) * length of atomic feature of each neighbor (z))
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        # atom_nbr_fea is a 3 dimensional matrix (number of atoms (x) * number of neighbors (y) * atomic feature of each neighbor (z))
        # nbr_fea is a 3 dimensional matrix (number of atoms (x) * number of neighbors (y) * guassian distance bond feature vector (z))
        # atom_in_fea is initially a 2 dimensional matrix.
        # It is converted to a 3 dimensional matrix by adding a new dimension in the y direction (1).
        # In this way, the atom feature vectors are repeated in the y-direction.
        # This is necessary for concatenation and forming final atom-bond-atom feature vector for each atom in the crystal.
        # total_nbr_fea is the concatenation of the 3 matrices which also a 3 dimensional matrix (dim = 2).
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        # total_gated_fea is the inner product calculation in eq.5.
        # total_gated_fea is a 3 dimensional matrix (number of atoms (x) * number of neighbors (y) * (2*atom_fea_len) (z))
        # This is becuase fc.full is defined this way manually above.
        total_gated_fea = self.fc_full(total_nbr_fea)
        # Batch normalization is performed (feature - mean/standard deviation)
        # For this, feature vector is reshaped first so that normalization occurs.
        # Then, the vector is reshaped back to its original dimension after normalizing.
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        # The 3 dimensional vector is split into 2 along the z direction.
        # Each chunk will have (number of atoms (x) * number of neighbors (y) * atom_fea_len (z) dimension.
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        # Sigmoid and softplus functions are applied on each of the chunks as per eq.5.
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        # Multiply nbr_filter & nbr_core as stated in eq.5
        # Sum across all neighbors for each atom (y-direction denoted by dim = 1)
        # nbr_sumed becomes a 2 dimensional matrix (number of atoms (x) * atom feature vector (y))
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        # Secondary normalization for the same atom features along y direction of nbr_sumed matrix.
        nbr_sumed = self.bn2(nbr_sumed)
        # The final step in eq.5 is summing the new matrix with the original atomoic feature matrix.
        # Hence, out becomes the updated atomic feature matrix for the crystal (output of Eq.5)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


# In[ ]:


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,classification=False):
                 
        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        # The length of original atom feature vector is reduced from 92 to 64 before passsing the convolution layer.
        # The function for transformation is linear (y = Wx + b)
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # Storing n_conv number of convolution layers in self.convs
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        # The input matrix from the convolution layer to the fully connected layer
        # after pooling has a dimension number of crystals * length of atom feature vector length.
        
        # This is the first hidden layer with h_fea_len hidden units.
        self.conv_to_fc = nn.Linear(atom_fea_len , h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        # If number of hidden layers is more than 1, then we will construct
        # the remaining layers.
        if n_h > 1:
            # y = Wx + b
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            # y = g(y)
            self.sigmoids = nn.ModuleList([nn.Sigmoid()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2) # For classification, there will be 2 output units.
        else:
            self.fc_out = nn.Linear(h_fea_len, 1) # For regression, there will only be 3 output units.
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch
        Parameters
        Z: Number of additional crystal features
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        crystal_fea: torch.LongTensor shape (N0, Z)
        
        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'sigmoids'):
            for fc, sigmoid in zip(self.fcs, self.sigmoids):
                crys_fea = sigmoid(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features
        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        # crystal_atom_idx is a list of tensors such that the length of tensor
        # is equal to the number of atoms in each crystal.
        # atom_fea variable is the atom features of a collection of crystals.
        # Hence, the number of rows is equal to the total number of atoms in all crystals.
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==            atom_fea.data.shape[0]
        
        # Each idx_map represents a numpy array containing the consecutive indicies representing the
        # atomic features for each distinct crystal. In this way, only the mean of features 
        # corresponding to each crystal is used to calculate the mean. Hence, batch processing works
        # fine.
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

