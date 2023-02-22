import math
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

new_size = 80

class SparseGraphAttentionLayer(torch.nn.Module):
  def __init__(self, in_features, out_features, dropout, leaky_relu_negative_slope = 0.2):# previously 0
    super(SparseGraphAttentionLayer, self).__init__()
    self.n_hidden = out_features
    self.linear = nn.Linear(in_features, self.n_hidden, bias=False)
    self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
    self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
    self.dropout = dropout
    self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) 
    self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1))) 
    self.alpha = leaky_relu_negative_slope


  def forward(self, h, edge_list):

    x = F.dropout(h, self.dropout, training=self.training)
    #sparse
    # h = torch.matmul(x, self.weight)
    #gat
    h = self.linear(h)
    source, target = edge_list
    a_input = torch.cat([h[source], h[target]], dim=1)
    #sparse  
    # e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)
    #gat
    e = self.activation(self.attn(a_input))
    # attention = sp_softmax(edge_list, e, h.size(0))
    # attention = F.dropout(attention, self.dropout, training=self.training)
    
    return e
    
        
class Pearson(torch.nn.Module):
    def __init__(self, region):
        
        super(Pearson, self).__init__()
        self.region = region
    def forward(self, chicago_target_region_representation, batch_no, train_or_test):
        
        if train_or_test == 0:
            folder = 'train/'
        else:
            folder = 'test/'
        with open('ADSF_Representation/chicago/node_embedding/' + folder + str(batch_no) + '/feature_representation_one_row_per_precinct_' + str(new_size) + '_0.3.pkl' , 'rb') as f:
            chicago_rep = pickle.load(f)
        chicago_target_region_rep = chicago_target_region_representation
            
        correlation_values = []
        x = chicago_target_region_rep
        for i in range(77):
            y = chicago_rep[i]
            region_representations = []
            region_representations.append(list(x))
            region_representations.append(list(y))
            input_matrix = torch.tensor(region_representations) # added
            correlation = torch.corrcoef(input_matrix)
            correlation_values.append(sys.float_info.min if math.isnan(correlation[0][1]) else correlation[0][1])
        return correlation_values


# class Jaccard(torch.nn.Module):
#     def __init__(self, region):
    
#         super(Jaccard, self).__init__()
#         self.region = region
#     def forward(self, chicago_target_region_representation):
        
#         chicago_target_region_rep = chicago_target_region_representation
        
#         with open('ADSF_Representation/chicago/node_embedding/feature_representation_one_row_per_precinct_120_0.3.pkl' , 'rb') as f:
#             chicago_rep = pickle.load(f)
            
#         correlation_values = []
#         y_true = np.array(chicago_target_region_rep.tolist())
#         for i in range(77):
#             y_pred = np.array(chicago_rep[i].tolist())
#             correlation = jaccard_score(y_true, y_pred, average="micro")
#             correlation_values.append(sys.float_info.min if math.isnan(correlation) else correlation)
#         return correlation_values
    
    
class Similarity(torch.nn.Module):
    def __init__(self, region):

        super(Similarity, self).__init__()
        self.region = region
        self.in_features = new_size
        self.out_features = new_size + 10
        self.dropout = 0.3
        self.layer1 = SparseGraphAttentionLayer(self.in_features, self.out_features, self.dropout)
        
            
            
    def forward(self, chicago_region_representations):
        
        chicago_target_region_temp = chicago_region_representations[self.region]
        chicago_temp = chicago_region_representations
        region_representations = []
        region_representations.append(list(chicago_target_region_temp))
        for i in range(77):
            region_representations.append(list(chicago_temp[i]))
        input_matrix = torch.tensor(region_representations) # added
        
        feature_tensor = input_matrix
        feature_tensor = feature_tensor.type(torch.FloatTensor)
        edge_list = []
        one= []
        two = []
        for j in range(77):
            one.append(0)
        for j in range(1, 78):
            two.append(j)
        edge_list.append(one)
        edge_list.append(two)
        
        attentions = self.layer1(feature_tensor, edge_list)
        return attentions # returns a 77x1 sized torch with similarity score for each region of chicago


# model  = Similarity(10)
# print("Similarity model : ", model)
# n = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Parameter  simi: ", n)