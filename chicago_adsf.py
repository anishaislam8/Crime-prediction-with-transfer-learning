import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

new_size = 80

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


#original code utils_nhop_neighbours.py

def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the structural fingerprints for nyc"""
    for i in range(len(ri_index)):
        for j in range(len(ri_index)):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            g[i][j] = 0
            if len(intersection) == 0:
                g[i][j] = 0.0001
                break
            else:
                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num

    return g


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, is_concat = True, dropout = 0.3, leaky_relu_negative_slope = 0.2): # previously negative slope 0,2
        super(GraphAttentionLayer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
    
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj_mat, s):
        n_nodes = h.shape[0] # 77
    
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)
        # assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        # assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        # assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf'))
        s = s.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        s_prime = self.softmax(s)
        new_attention = self.softmax(a.add(s_prime))
        new_attention = self.dropout(new_attention)
        attn_res = torch.einsum('ijh,jhf->ihf', new_attention, g)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else :
            return attn_res.mean(dim=1)
        
      

def preprocessing():
    
    f = open("chicago_data/chicago/com_adj_matrix.txt", "r")
    lines = f.readlines()
    
    adj_list = []
    final_adj_list = []
    for line in lines :
      current_array = line.split(" ")
      desired_array = [int(numeric_string) for numeric_string in current_array]
      adj_list.append(desired_array)
    
    for i in range(len(adj_list)):
      arr = np.array(adj_list[i])
      desired_array = np.where(arr == 1)
      final_adj_list.append(desired_array[0].tolist())
    
    f.close()
    
    f = open("chicago_data/chicago/com_adj_matrix_list.txt", "w")
    for i in range(len(final_adj_list)):
      for j in range(len(final_adj_list[i])):
        f.write(str(final_adj_list[i][j]))
        if j != len(final_adj_list[i]) - 1 :
          f.write(" ")
      if i != len(final_adj_list) - 1 :
        f.write("\n")
    f.close()
    
    #create a dictionary of adjacency lists

    graph = {}
    i = 0
    for elem in final_adj_list:
        graph[i] = elem
        i+=1
    
    
    # Original file utils_nhop_neighbours.py line 157 - 182

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    adj_delta = adj
    
    # caculate n-hop neighbors
    G = nx.Graph() # use normal graph
    
    for i in range(len(final_adj_list)):
        for j in range(len(final_adj_list[i])):
          G.add_edge(i, final_adj_list[i][j], weight=1)
    for i in range(len(final_adj_list)):
        for j in range(len(final_adj_list)):
            try:
                rs = nx.astar_path_length \
                        (
                        G,
                        i,
                        j,
                    )
            except nx.NetworkXNoPath:
                rs = 0
            if rs == 0:
                length = 0
            else:
                #print(rs)
                length = rs
            adj_delta[i][j] = length
    
    #Original file rwr_process.py line 35 - 92
    dijkstra = adj_delta
    
    Dijkstra = dijkstra.numpy()
    ri_all = []
    ri_index = []
    # You may replace 3327 with the size of dataset
    for i in range(len(final_adj_list)):
        # You may replace 1,4 with the .n-hop neighbors you want
        index_i = np.where((Dijkstra[i] < 4) & (Dijkstra[i] > 1))
        I = np.eye((len(index_i[0]) + 1), dtype=int)
        ei = []
        for q in range((len(index_i[0]) + 1)):
            if q == 0:
                ei.append([1])
            else:
                ei.append([0])
        W = []
        for j in range((len(index_i[0])) + 1):
            w = []
            for k in range((len(index_i[0])) + 1):
                if j == 0:
                    if k == 0:
                        w.append(float(0))
                    else:
                        w.append(float(1))
                else:
                    if k == 0:
                        w.append(float(1))
                    else:
                        w.append(float(0))
            W.append(w)
        # the choice of the c parameter in RWR
        c = 0.5
        W = np.array(W)
        rw_left = (I - c * W)
        try:
            rw_left = np.linalg.inv(rw_left)
        except:
            rw_left = rw_left
        else:
            rw_left = rw_left
        ei = np.array(ei)
        rw_left = torch.tensor(rw_left, dtype=torch.float32) # added
        ei = torch.tensor(ei, dtype=torch.float32) # added
        ri = torch.mm(rw_left, ei)
        ri = torch.transpose(ri, 1, 0)
        ri = abs(ri[0]).numpy().tolist()
        ri_index.append(index_i[0])
        ri_all.append(ri)
   
    
    # Evaluate structural interaction between the structural fingerprints of node i and j
    adj_delta = structural_interaction(ri_index, ri_all, adj_delta)

    f = open("chicago_data/chicago/com_adj_matrix.txt", "r")
    lines = f.readlines()
    chicago_adjacency_matrix = []
    for line in lines :
        current_array = line.split(" ")
        desired_array = [int(numeric_string) for numeric_string in current_array]
        chicago_adjacency_matrix.append(desired_array)
    chicago_adjacency_matrix = np.array(chicago_adjacency_matrix)
    return adj_delta, chicago_adjacency_matrix
     
class ADSF(torch.nn.Module):
    
    def __init__(self, prec):
        super(ADSF, self).__init__()
        self.prec = prec
        self.in_features = None
        self.out_features = new_size
        self.n_heads = 1
        self.dropout = 0.3
        self.is_concat = True
        self.layer1 = None

    def forward(self, road_representation, poi_representation, inflows, outflows, batch_no):
        
        structural_interaction, chicago_adjacency_matrix = preprocessing()
        structural_interaction = structural_interaction.resize_((structural_interaction.shape[0], structural_interaction.shape[1], 1))
        chicago_adjacency_matrix = torch.tensor(chicago_adjacency_matrix)
        chicago_adjacency_matrix = chicago_adjacency_matrix.resize_((chicago_adjacency_matrix.shape[0], chicago_adjacency_matrix.shape[1], 1))
        road_network = road_representation
        poi = poi_representation
 
        feature_matrix = []
        for i in range(77):
            result = torch.vstack((road_network[i]*0.167, poi[i]*0.501, inflows[i][batch_no]*0.166, outflows[i][batch_no]*0.166))
            feature_matrix.append(result)
        
        biggest = 0
        for i in range(77):
          if feature_matrix[i].shape[0] * feature_matrix[i].shape[1] > biggest:
            biggest = feature_matrix[i].shape[0] * feature_matrix[i].shape[1]
            
            
        new_matrix = []
        for i in range(77):
          temp = list(torch.reshape(feature_matrix[i], (feature_matrix[i].shape[0] * feature_matrix[i].shape[1],)))
          avg_temp = sum(temp)/len(temp)
          if len(temp) < biggest:
            limit= len(temp)
            for m in range(biggest-limit):
              temp.append(avg_temp)
          new_matrix.append(temp)
        new_matrix = torch.tensor(new_matrix).type(torch.FloatTensor) # added
        self.in_features = new_matrix.shape[1]
        
        if self.layer1 == None:
            self.layer1 = GraphAttentionLayer(self.in_features, self.out_features, self.n_heads, self.is_concat, self.dropout)
       
        feature_tensor = new_matrix
        feature_tensor = feature_tensor.type(torch.FloatTensor)
        feature_representation = self.layer1(feature_tensor, chicago_adjacency_matrix.float(), structural_interaction.float())
        return feature_representation
  
  
# model =ADSF(10)
# print("ADSF: ", model)
# n = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Parameter  adsf: ", n)

    
    
    
