import numpy as np 
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

class GCNLayer(nn.Module):
    """
        GCN layer

        Args:
            input_dim (int): Dimension of the input
            output_dim (int): Dimension of the output (a softmax distribution)
            A (torch.Tensor): 2D adjacency matrix
    """

    def __init__(self, input_dim: int, output_dim: int, nodes: int):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nodes = nodes
        
        # Initialise the weight matrix as a parameter
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        # X shape: (batch_size, num_nodes, input_dim)
        # A shape: (batch_size, num_nodes, num_nodes)

        batch_size = X.size(0)
        num_nodes = X.size(1)
        
        # Add identity matrix I to A (A_hat = A + I)
        I = torch.eye(num_nodes, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
        A_hat = A + torch.max(A)*I  # A_hat shape: (batch_size, num_nodes, num_nodes)

        # Create diagonal degree matrix D
        D = torch.sum(A_hat, dim=-1)  # D shape: (batch_size, num_nodes)
        D = torch.diag_embed(D)  # D shape: (batch_size, num_nodes, num_nodes)

        # Create D^{-1/2}
        D_neg_sqrt = torch.diag_embed(torch.diagonal(D, dim1=1, dim2=2).pow(-0.5))  # D_neg_sqrt shape: (batch_size, num_nodes, num_nodes)
        
        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(D_neg_sqrt, torch.matmul(A_hat, D_neg_sqrt))  # support_1 shape: (batch_size, num_nodes, num_nodes)
        
        # (D^-1/2 * A_hat * D^-1/2) * (X * W)
        support_2 = torch.matmul(support_1, torch.matmul(X, self.W))  # support_2 shape: (batch_size, num_nodes, output_dim)
        
        # ReLU(D^-1/2 * A_hat * D^-1/2 * X * W)
        H = F.relu(support_2)

        return H

class GYMnet(nn.Module):
    def __init__(self, 
                 nodes_1: int, 
                 nodes_2: int, 
                 nclass: int, 
                 feats_lin: int = 32,
                 dropout: float=0.1,
                 nfeats: int=384, 
                 dmax: int=20): 
        super(GYMnet, self).__init__()

        self.graph_nodes_1 = nodes_1
        self.graph_nodes_2 = nodes_2
        self.nclass = nclass

        self.linear = nn.Linear(4*nodes_2, feats_lin)
        self.linear_2 = nn.Linear(feats_lin, nclass)
        self.gcn_1 = GCNLayer(nfeats, self.graph_nodes_1, dmax)
        self.gcn_2 = GCNLayer(self.graph_nodes_1, self.graph_nodes_2, dmax)

        self.drop = nn.Dropout(p=dropout)

    def forward(self, X, A):

        # Create A_ matrix to be able to average over sites
        A_ = (A == 1).float()
        I = torch.eye(A_.size(1), device=A.device).unsqueeze(0).expand(X.size(0), -1, -1)
        A_ = A_ + I  # A_hat shape: (batch_size, num_nodes, num_nodes)
        A_ = A_[:, :4] # cut A based on 4 modalities 

        # first gcn layer
        X = self.gcn_1(X, A)
                
        # add dropout layer 
        X = self.drop(X)

        # second gcn layer
        X = self.gcn_2(X, A)
                
        # add dropout layer 
        X = self.drop(X)

        X = torch.matmul(A_, X) / A_.sum(dim=2).unsqueeze(2)

        X = torch.flatten(X, start_dim=1)

        # send it through a dense layer 
        X = self.linear(X)
        
        # add dropout layer 
        X = self.drop(X)

        X = self.linear_2(X)

        # apply sigmoid activation
        #X = nn.Sigmoid()(X)
        return X

# predictions should be done by well
class GYMData(torch.utils.data.Dataset):
    def __init__(self, 
                 feats: np.array, 
                 meta: pd.DataFrame, 
                 nmoa: int, 
                 l: int=2, 
                 dmax: int=20
                 ):
        '''
        Args:
            feats (np.array): features
            meta (pd.DataFrame): meta data
            l (int): lambda value for the adjacency matrix
            dmax (int): maximum number of data samples per well
            nmoa (int): number of MoA classes
        '''
        self.feats = feats
        self.meta = meta
        self.l = l
        self.dmax = dmax
        self.nmoa = nmoa

        indices = meta.groupby(['well', 'plate_name', 'moa_coded']).groups

        self.moas = np.asarray(list(indices.keys()))[:, 2]
        self.moas = self.moas.astype(int)
        self.indices = list(indices.values())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        X = torch.tensor(self.feats[self.indices[idx]])
        A = self.get_adj_matrix(self.feats[self.indices[idx]], self.meta.iloc[self.indices[idx]], self.l)
        moa = self.moas[idx]

        #moa = F.one_hot(torch.tensor(int(moa)), self.nmoa).float()

        # pad the values for the consistent input
        if X.shape[0] != self.dmax: 
            X = torch.cat((X, torch.zeros(self.dmax - X.shape[0], X.shape[1])))
            A = F.pad(A, (0, self.dmax - A.shape[0], 0, self.dmax - A.shape[1]))
        return {'feats' : X, 'adj_matrix': A, 'moa': torch.tensor(moa).type(torch.LongTensor) }
    
    def get_adj_matrix(self, feats, meta, l=2):
        '''
        Create adjecency matrix for the graph.
        '''
        assert feats.shape[0] == len(meta) # Check if the number of features is equal to the number of meta data

        meta = meta.reset_index(drop=True)
        
        A = torch.zeros((len(meta), len(meta)))

        modalities = meta.modality_name.unique()
        sites = meta.site.unique()

        #populate the adjacency matrix for each modality and each site
        for m in modalities: # connecting between sites 
            ind = meta.query('modality_name == @m').index
            ind = np.array(list(product(ind, ind)))

            A[ind[:, 0], ind[:, 1]] = 2
        
        for s in sites: # connecting between modalities 
            ind = meta.query('site == @s').index
            ind = np.array(list(product(ind, ind)))

            A[ind[:, 0], ind[:, 1]] = l
        
        # remove self loops (they are already included in the GCNLayer)
        A-=l*torch.eye(A.size(0))

        return A