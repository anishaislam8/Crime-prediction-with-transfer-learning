import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit

transform = RandomLinkSplit(is_undirected=True)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning # previously 2 instead of 100
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning # previously 2 instead of 100

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class RoadNetwork(torch.nn.Module):
    def __init__(self, region):
        super(RoadNetwork, self).__init__()
        with open('chicago autoencoder inputs/chicago_x_intersection.pkl', 'rb') as handle:
          self.x = pickle.load(handle)
        with open('chicago autoencoder inputs/chicago_edge_index_0_indexed.pkl', 'rb') as handle:
          self.edge_index = pickle.load(handle)
        
        self.region = region
        self.num_features = self.x[self.region].shape[1]
        self.out_channels = 10
        self.autoencoder = GAE(GCNEncoder(self.num_features, self.out_channels))
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.01)
        
    def forward(self):

        data = Data(num_nodes = self.x[self.region].shape[0], edge_index=torch.tensor(self.edge_index[self.region]), train_mask = None, test_mask = None, val_mask = None)
        train_data, val_data, test_data = transform(data)
        train_pos_edge_index = train_data.edge_label_index
        x_data = torch.tensor(self.x[self.region], dtype=torch.float)
        
        Z = self.autoencoder.encode(x_data, train_pos_edge_index)
        return Z
        
class PointsOfInterest(torch.nn.Module):
    
    def __init__(self, region):
        super(PointsOfInterest, self).__init__()
        with open('chicago autoencoder inputs/chicago_poi_x.pkl', 'rb') as handle:
          self.x = pickle.load(handle)
        with open('chicago autoencoder inputs/chicago_poi_edge_index_0_indexed.pkl', 'rb') as handle:
          self.edge_index = pickle.load(handle)
        self.region = region
        self.num_features = self.x[self.region].shape[1]
        self.out_channels = 10
        self.autoencoder = GAE(GCNEncoder(self.num_features, self.out_channels))
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.01)
        
    def forward(self):
        data = Data(num_nodes = self.x[self.region].shape[0]+1, edge_index=torch.tensor(self.edge_index[self.region]), train_mask = None, test_mask = None, val_mask = None)
        train_data, val_data, test_data = transform(data)
        train_pos_edge_index = train_data.edge_label_index
        x_data = torch.tensor(self.x[self.region], dtype=torch.float)
       
        Z = self.autoencoder.encode(x_data, train_pos_edge_index)
        return Z