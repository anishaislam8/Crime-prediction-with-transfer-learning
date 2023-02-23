import torch
import torch.nn as nn
from torch.autograd import Variable

from chicago_adsf import *
from chicago_feature_representation import *
from layers import *
from similarity_measurement import *
from utils import *

new_size = 4
dropout_val = 0.3


class ModifiedAIST(nn.Module):
    def __init__(self, region, in_hgat, in_fgat, out_gat, att_dot, nhid_rnn, nlayer_rnn, att_rnn, ts, batch_size, nclass=2):
        """
        :param in_hgat: dimension of the input of hgat
        :param in_fgat: dimension of the input of fgat
        :param out_gat: dimension of the output of gat
        :param att_dot: dimension of the dot attention of gat
        :param nhid_rnn: dimension of hidden state of rnn
        :param nlayer_rnn: number of layers of rnn
        :param att_rnn: dimension of attention of trend
        :param att_rnn: number of time steps
        :param target_region: starts with 0
        :param target_cat: starts with 0
        """
        super(ModifiedAIST, self).__init__()
        self.in_hgat = in_hgat
        self.in_fgat = in_fgat
        self.out_gat = out_gat
        self.att_dot = att_dot
        self.nhid_rnn = nhid_rnn
        self.nlayer_rnn = nlayer_rnn
        self.att_rnn = att_rnn
        self.batch_size = batch_size

        self.sab1 = self_LSTM_sparse_attn_predict(
            out_gat+new_size, nhid_rnn, nlayer_rnn, truncate_length=5, top_k=4, attn_every_k=5)
        self.fc1 = nn.Linear(nhid_rnn, 1)
        self.fc2 = nn.Linear(2 * nhid_rnn, 1)
        self.fc3 = nn.Linear(nhid_rnn, 1)
        # parameters for trend-attention
        self.wv = nn.Linear(nhid_rnn, self.att_rnn)  # (S, E) x (E, 1) = (S, 1)
        # attention of the trends
        self.wu = nn.Parameter(torch.zeros(size=((42), self.att_rnn)))
        nn.init.xavier_uniform_(self.wu.data, gain=1.414)

        self.dropout_layer = nn.Dropout(p=dropout_val)  # previously 0.2

        self.roadNetworkRepresentation = []
        for i in range(77):
            self.roadNetworkRepresentation.append(RoadNetwork(i))
        self.poiRepresentation = []
        for i in range(77):
            self.poiRepresentation.append(PointsOfInterest(i))

        self.adsf = ADSF(region)
        self.similarity = Similarity(region)
        # self.similarity = Pearson(region)

    def forward(self, region, batch, batch_size, similar_region_number, target_cat_chicago, loaded_data, train_or_test, current_crime_category, inflows, outflows):

        # Feature Representation using Auto-encoder + ADSF
        road_representation = []
        for i in range(77):
            road_representation.append(self.roadNetworkRepresentation[i]())
        poi_representation = []
        for i in range(77):
            poi_representation.append(self.poiRepresentation[i]())

        region_feature_representations = self.adsf(
            road_representation, poi_representation, inflows, outflows, batch)  # passing 77 values
        # print("Feature representation : ", region_feature_representations[region][:10])

        # Similarity score calculation
        # using gat
        attention_values = self.similarity(region_feature_representations)
        # print(attention_values)
        # using pearson coefficient
        # attention_values = torch.tensor(self.similarity(region_feature_representations[region], batch, train_or_test))
        # print(attention_values)

        # best similar region selection
        attention_values = attention_values.flatten()  # 1D array of shape (77,)
        similar_regions = torch.argsort(-attention_values)
        similar_regions = similar_regions.tolist()
        similar_regions.remove(region)
        similar_regions = similar_regions[:similar_region_number]
        # print("Similar regions : ", similar_regions)

        attention_list = []
        for att in range(similar_region_number):
            attention_list.append(attention_values[similar_regions[att]])
        minimum = min(attention_list)
        maximum = max(attention_list)

        denominator = False
        if maximum == minimum:
            denominator = True

        normalized_attention_list = []
        for att in range(similar_region_number):
            if denominator == False:
                val = (
                    ((attention_values[similar_regions[att]] - minimum)*1.0)/((maximum - minimum)*1.0))
            else:
                val = (
                    ((attention_values[similar_regions[att]] - minimum)*1.0)/(0.00001))

            if val == 0:
                val = 0.00001
            normalized_attention_list.append(val)

        for att in range(similar_region_number):
            attention_values[similar_regions[att]
                             ] = normalized_attention_list[att]
            # print("Attention value for " + str(similar_regions[att]) + " : " + str(attention_values[similar_regions[att]]))

        # get similar region crime summation
        train_y_tl, test_y_tl = similar_region_crimes(
            loaded_data, similar_regions, similar_region_number, target_cat_chicago)
        if train_or_test == 0:
            y = Variable(train_y_tl[batch]).float()
            y = y.view(-1, 1)
        else:
            y = Variable(test_y_tl[batch]).float()
            y = y.view(-1, 1)

        # get crime representation from AIST
        final_crime_representation_for_this_batch = crime_representation(
            train_or_test, similar_regions, attention_values, current_crime_category, similar_region_number, batch)
        feature = region_feature_representations[region].reshape(
            (20, new_size))
        temp = []
        for j in range(batch_size):
            temp.append(torch.reshape(
                feature, (1, feature.shape[0], feature.shape[1])))
        feature = torch.empty(batch_size, feature.shape[0], feature.shape[1])
        feature = torch.cat(temp)
        lstm_input = torch.concat(
            (final_crime_representation_for_this_batch, feature), axis=2)

        x_con, x_con_attn = self.sab1(lstm_input)  # x_con = (B, ts)
        x_con = self.dropout_layer(x_con)
        x = torch.tanh(self.fc1(x_con))

        return x, y
