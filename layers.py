import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tensorflow as tf


class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):

        attn_plot = []
        eps = 10e-8
        batch_size = attn_s.size()[0]
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            return attn_s
        else:
            delta = torch.topk(attn_s, self.top_k, dim=1)[
                0][:, -1] + eps  # updated myself

        attn_w = attn_s - delta.reshape((batch_size, 1)).repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / \
            attn_w_sum.reshape((batch_size, 1)).repeat(1, time_step)
        return attn_w_normalize


class self_LSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2,
                 truncate_length=100, block_attn_grad_past=False, attn_every_k=1, top_k=5):
        """
        :param input_size: number of features at each time step
        :param hidden_size: dimension of the hidden state of the lstm
        :param num_layers: number of layers of the lstm
        :return attn_c: output of sab-lstm
        :return out_attn_w: attention state of sab-lstm

        """
        # latest sparse attentive back-prop implementation
        super(self_LSTM_sparse_attn_predict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)

        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()

        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.w_t.data, gain=1.414)

        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.predict_m = nn.Linear(hidden_size, 2)

    def forward(self, x):

        # x = x.view(x.shape[0], int(x.shape[1]/self.input_size), self.input_size)
        batch_size = x.size(0)
        time_size = x.size(1)  # num of time steps

        # h_t = (batch_size, hidden_size)
        h_t = Variable(torch.zeros(batch_size, self.hidden_size))
        # c_t = (batch_size, hidden_size)
        c_t = Variable(torch.zeros(batch_size, self.hidden_size))
        # predict_h = (batch_size, hidden_size)
        predict_h = Variable(torch.zeros(batch_size, self.hidden_size))

        # Will eventually grow to (batch_size, time_size, hidden_size) with more and more concatenations.
        # h_old = (batch_size, 1, hidden_size) --> Memory
        h_old = h_t.view(batch_size, 1, self.hidden_size)

        outputs = []
        attn_all = []
        attn_w_viz = []
        predicted_all = []
        outputs_new = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            # input_t = (batch_size, 1, input_size)
            remember_size = h_old.size(1)

            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed LSTM Cell
            # input_t = (batch_size, input_size)
            input_t = input_t.contiguous().view(batch_size, self.input_size)
            # h_t/ c_t = (batch_size, hidden dimension)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            predict_h = self.predict_m(h_t.detach())
            predicted_all.append(h_t)  # changed predict_h

            # Broadcast and concatenate current hidden state against old states
            # h_repeated = (batch_size, remember_size = memory, hidden_size)
            h_repeated = h_t.unsqueeze(1).repeat(1, remember_size, 1)
            # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)
            mlp_h_attn = self.tanh(mlp_h_attn)

            if False:  # PyTorch 0.2.0
                attn_w = torch.matmul(mlp_h_attn, self.w_t)
            else:  # PyTorch 0.1.12
                # mlp_h_attn = (batch_size * remember_size, 2* hidden_size)
                mlp_h_attn = mlp_h_attn.view(
                    batch_size * remember_size, 2 * self.hidden_size)
                # attn_w = (batch_size * remember_size, 1)
                attn_w = torch.mm(mlp_h_attn, self.w_t)
                # attn_w = (batch_size, remember_size, 1)
                attn_w = attn_w.view(batch_size, remember_size, 1)
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            # attn_w = (batch_size, remember_size)
            attn_w = attn_w.view(batch_size, remember_size)
            # attn_w = (batch_size, remember_size)
            attn_w = self.sparse_attn(attn_w)
            # attn_w = (batch_size, remember_size, 1)
            attn_w = attn_w.view(batch_size, remember_size, 1)

            # if i >= 100:
            # print(attn_w.mean(dim=0).view(remember_size))
            attn_w_viz.append(attn_w.mean(dim=0).view(
                remember_size))  # you should return it
            out_attn_w = attn_w
            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            # attn_w = (batch_size, remember_size, hidden_size)
            attn_w = attn_w.repeat(1, 1, self.hidden_size)
            # attn_w = (batch_size, remember_size, hidden_size)
            h_old_w = attn_w * h_old
            attn_c = torch.sum(h_old_w, 1).squeeze(
                1)  # att_c = (batch_size, hidden_size)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c  # h_t = (batch_size, hidden_size)

            #
            # At regular intervals, remember a hidden state, store it in memory
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat(
                    (h_old, h_t.view(batch_size, 1, self.hidden_size)), dim=1)

            # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            predict_real_h_t = self.predict_m(h_t.detach())
            outputs_new += [predict_real_h_t]

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        # predicted_all = (batch_size, time_step, hidden_size)
        predicted_all = torch.stack(predicted_all, 1)
        # outputs = (batch_size, time_step, hidden_size)
        outputs = torch.stack(outputs, 1)
        # attn_all = (batch_size, time_step, hidden_size)
        attn_all = torch.stack(attn_all, 1)

        return attn_c, out_attn_w


# class self_LSTM_sparse_attn_predict_NYC(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes=2,
#                  truncate_length=100, block_attn_grad_past=False, attn_every_k=1, top_k=5):
#         """
#         :param input_size: number of features at each time step
#         :param hidden_size: dimension of the hidden state of the lstm
#         :param num_layers: number of layers of the lstm
#         :return attn_c: output of sab-lstm
#         :return out_attn_w: attention state of sab-lstm

#         """
#         # latest sparse attentive back-prop implementation
#         super(self_LSTM_sparse_attn_predict_NYC, self).__init__()
#         self.input_size = input_size # 58
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_classes = num_classes
#         self.lstm1 = nn.LSTMCell(input_size, hidden_size)


#         self.block_attn_grad_past = block_attn_grad_past
#         self.truncate_length = truncate_length
#         self.fc = nn.Linear(hidden_size * 2, num_classes)
#         self.fc1 = nn.Linear(hidden_size, num_classes)
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#         self.attn_every_k = attn_every_k
#         self.top_k = top_k
#         self.tanh = torch.nn.Tanh()

#         self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
#         nn.init.xavier_uniform_(self.w_t.data, gain=1.414)

#         self.sparse_attn = Sparse_attention(top_k=self.top_k)
#         self.predict_m = nn.Linear(hidden_size, 2)

#     def forward(self, x):

#         # print(x.shape)
#         # x = x.view(x.shape[0], int(x.shape[1]/self.input_size), self.input_size)
#         batch_size = x.size(0) # (42) -> 1
#         time_size = x.size(1)  # num of time steps,20 -> 1160


#         h_t = Variable(torch.zeros(batch_size, self.hidden_size))  # h_t = (batch_size, hidden_size)
#         c_t = Variable(torch.zeros(batch_size, self.hidden_size))  # c_t = (batch_size, hidden_size)
#         predict_h = Variable(torch.zeros(batch_size, self.hidden_size))  # predict_h = (batch_size, hidden_size)

#         # Will eventually grow to (batch_size, time_size, hidden_size) with more and more concatenations.
#         h_old = h_t.view(batch_size, 1, self.hidden_size)  # h_old = (batch_size, 1, hidden_size) --> Memory

#         outputs = []
#         attn_all = []
#         attn_w_viz = []
#         predicted_all = []
#         outputs_new = []

#         for i, input_t in enumerate(x.chunk(time_size, dim=1)):
#             # input_t = (batch_size, 1, input_size)
#             remember_size = h_old.size(1)
#             # print(input_t.shape)

#             if (i + 1) % self.truncate_length == 0:
#                 h_t, c_t = h_t.detach(), c_t.detach()

#             # Feed LSTM Cell
#             input_t = input_t.contiguous().view(batch_size, self.input_size)  # input_t = (batch_size, input_size)
#             h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # h_t/ c_t = (batch_size, hidden dimension)
#             predict_h = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
#             predicted_all.append(h_t)  # changed predict_h

#             # Broadcast and concatenate current hidden state against old states
#             h_repeated = h_t.unsqueeze(1).repeat(1, remember_size, 1)  # h_repeated = (batch_size, remember_size = memory, hidden_size)
#             mlp_h_attn = torch.cat((h_repeated, h_old), 2)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

#             if self.block_attn_grad_past:
#                 mlp_h_attn = mlp_h_attn.detach()

#             #
#             # Feed the concatenation to the MLP.
#             # The tensor shapes being multiplied are
#             #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
#             # by
#             #     self.w_t.size()   = (2*hidden_size, 1)
#             # Desired result is
#             #     attn_w.size()     = (batch_size, remember_size, 1)
#             #
#             mlp_h_attn = self.tanh(mlp_h_attn)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

#             if False:  # PyTorch 0.2.0
#                 attn_w = torch.matmul(mlp_h_attn, self.w_t)
#             else:  # PyTorch 0.1.12
#                 mlp_h_attn = mlp_h_attn.view(batch_size * remember_size, 2 * self.hidden_size)  # mlp_h_attn = (batch_size * remember_size, 2* hidden_size)
#                 attn_w = torch.mm(mlp_h_attn, self.w_t)  # attn_w = (batch_size * remember_size, 1)
#                 attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)
#             #
#             # For each batch example, "select" top-k elements by sparsifying
#             # attn_w.size() = (batch_size, remember_size, 1). The top k elements
#             # are left non-zero and the other ones are zeroed.
#             #
#             attn_w = attn_w.view(batch_size, remember_size)  # attn_w = (batch_size, remember_size)
#             attn_w = self.sparse_attn(attn_w)  # attn_w = (batch_size, remember_size)
#             attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)

#             # if i >= 100:
#             # print(attn_w.mean(dim=0).view(remember_size))
#             attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))  # you should return it
#             out_attn_w = attn_w
#             #
#             # Broadcast the weights against the past remembered hidden states,
#             # then compute the attention information attn_c.
#             #
#             attn_w = attn_w.repeat(1, 1, self.hidden_size)  # attn_w = (batch_size, remember_size, hidden_size)
#             h_old_w = attn_w * h_old  # attn_w = (batch_size, remember_size, hidden_size)
#             attn_c = torch.sum(h_old_w, 1).squeeze(1)  # att_c = (batch_size, hidden_size)

#             # Feed attn_c to hidden state h_t
#             h_t = h_t + attn_c  # h_t = (batch_size, hidden_size)

#             #
#             # At regular intervals, remember a hidden state, store it in memory
#             #
#             if (i + 1) % self.attn_every_k == 0:
#                 h_old = torch.cat((h_old, h_t.view(batch_size, 1, self.hidden_size)), dim=1)

#             predict_real_h_t = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
#             outputs_new += [predict_real_h_t]

#             # Record outputs
#             outputs += [h_t]

#             # For visualization purposes:
#             attn_all += [attn_c]

#         #
#         # Compute return values. These should be:
#         #     out        = (batch_size, time_size, num_classes)
#         #     attn_w_viz = len([(remember_size)]) == time_size-100
#         #
#         predicted_all = torch.stack(predicted_all, 1)  # predicted_all = (batch_size, time_step, hidden_size)
#         outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
#         attn_all = torch.stack(attn_all, 1)  # attn_all = (batch_size, time_step, hidden_size)

#         return attn_c, out_attn_w


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features_hgat, in_features_fgat, out_features, att_dot, target_region, target_cat, dropout, alpha, concat=True):
        """
        :param in_features_hgat: input dimension
        :param out_features: out dimension
        :param att_dot: dimension of the dot attention
        :return h_prime: crime representation of all the nodes
        :return ext_rep: feature representation of all the nodes
        """
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features_hgat = in_features_hgat
        self.in_features_fgat = in_features_fgat
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.target_region = target_region
        self.target_cat = target_cat

        self.W = nn.Parameter(torch.zeros(
            size=(in_features_hgat, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.Wf = nn.Parameter(torch.zeros(
            size=(in_features_hgat, out_features)))
        nn.init.xavier_uniform_(self.Wf.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.WS = nn.Parameter(torch.zeros(
            size=(in_features_hgat, out_features)))
        nn.init.xavier_uniform_(self.WS.data, gain=1.414)

        self.aS = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS.data, gain=1.414)

        self.WS1 = nn.Parameter(torch.zeros(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.WS1.data, gain=1.414)
        self.aS1 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS1.data, gain=1.414)

        self.att_dim = att_dot
        self.emb_dim = out_features
        self.nfeat = in_features_fgat

        self.WQ = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WQ.data, gain=1.414)

        self.WK = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WK.data, gain=1.414)

        self.WV = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.WV.data, gain=1.414)

        self.WF = nn.Linear(self.nfeat, out_features, bias=False)

    def forward(self, input, adj, ext_input, side_input):

        batch_size = (42)
        # shape = (B, N, 1) --> child crime input of each node (regions)
        input = input.view(batch_size, -1, 1)
        # shape = (B, N, 1) --> ext feat of each node
        ext_input = ext_input.view(batch_size, -1, self.nfeat)
        # shape = (B, N, 1) --> parent crime input of regions
        side_input = side_input.view(batch_size, -1, 1)
        adj = adj.repeat(batch_size, 1, 1)  # adj matrix

        """
            Find the attention vectors for 
            region_wise crime similarity
        """
        # Find the attention vectors for region_wise crime similarity
        h = torch.matmul(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[1]  # N = Number of Nodes (regions)
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(
            1, N, 1)], dim=2).view(h.shape[0], N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)  # shape = (B, N, N)
        attention = torch.where(adj > 0, e, zero_vec)  # shape = (B, N, N)

        # attention = F.softmax(attention, dim=2)  # shape = (B, N, N)
        attention = F.dropout(attention, self.dropout,
                              training=self.training)  # shape = (B, N, N)
        # h_prime = torch.matmul(attention, h)  # shape = (B, N, F'1)

        # Tensor shapes and co
        # h.repeat(1, 1, N).view(B, N * N, -1) = (B, NxN, F'), h.repeat(N, 1) = (B, NxN, F')
        # cat = (B, NxN, 2F')
        # a_input = (B, N, N, 2F')
        # torch.matmul(a_input, self.a).squeeze(2) = ((B, N, N, 1) -----> (B, N, N))

        """
            Find the attention vectors for 
            side_wise crime similarity
        """
        h_side = torch.matmul(
            side_input, self.WS)  # h = [h_1, h_2, h_3, ... , h_N] * W
        a_input_side = torch.cat([h_side.repeat(1, 1, N).view(
            (42), N * N, -1), h_side.repeat(1, N, 1)], dim=2).view((42), N, -1, 2 * self.out_features)
        e_side = self.leakyrelu(torch.matmul(a_input_side, self.aS).squeeze(3))
        attention_side = torch.where(
            adj > 0, e_side, zero_vec)  # shape = (B, N, N)
        attention_side = F.dropout(
            attention_side, self.dropout, training=self.training)  # shape = (B, N, N)
        # h_prime_side = torch.matmul(attention_side, h_side)  # shape = (B, N, F')

        """
            Find the crime representation of 
            a region
        """

        attention = attention + attention_side
        attention = torch.where(attention > 0, attention,
                                zero_vec)  # shape = (B, N, N)
        attention = F.softmax(attention, dim=2)  # shape = (B, N, N)
        attention = F.dropout(attention, self.dropout,
                              training=self.training)  # shape = (B, N, N)
        h_prime = torch.matmul(attention, h)  # shape = (B, N, F')

        """r_att = open("Heatmap/r_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        r_att_arr = attention[:, -1, :].mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(r_att, r_att_arr, fmt="%f")
        r_att.close()"""

        # # Generate Query
        # n_feature = self.in_features_fgat
        # q = torch.cat([input.repeat(1, 1, N).view(input.shape[0], N * N, -1), input.repeat(1, N, 1)], dim=2).view(input.shape[0], N, N, -1)
        # q = torch.matmul(q, self.WQ)  # (B, N, N, dq) = (B, N, N, 2) * (2, dq)
        # q = q / (self.att_dim ** 0.5)
        # q = q.unsqueeze(3)  # (B, N, N, 1, dq)

        # # Generate Key
        # # hf = self.WF(ext_input.unsqueeze(3))  # (B, N, nfeat, F') =
        # ext_input = ext_input.unsqueeze(3)
        # k = torch.cat([ext_input.repeat(1, 1, N, 1).view(ext_input.shape[0], N * N, n_feature, -1), ext_input.repeat(1, N, 1, 1).
        #               view(ext_input.shape[0], N * N, n_feature, -1)], dim=3).view(ext_input.shape[0], N, N, n_feature, 2)
        # k = torch.matmul(k, self.WK)  # (B, N, N, nfeat, dk) = (B, N, N, nfeat, 2)* (2, dk)
        # k = torch.transpose(k, 4, 3)  # (B, N, N, dk, nfeat)

        # # Generate Value
        # v = torch.matmul(ext_input, self.WV)  # (B, N, N, nfeat, dv)

        # # Generate dot product attention
        # dot_attention = torch.matmul(q, k).squeeze(3)  # (B, N, N, nfeat)
        # zero_vec = -9e15 * torch.ones_like(dot_attention)
        # dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (B, N, N, nfeat)
        # dot_attention = F.softmax(dot_attention, dim=3)  # shape = (B, N, N, nfeat)

        # """f_att = open("Heatmap/f_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        # f_att_arr = dot_attention.mean(dim=0)[-1].sum(dim=0).detach().softmax(dim=0).view(1, -1).numpy()
        # np.savetxt(f_att, f_att_arr, fmt="%f")
        # f_att.close()"""

        # """
        #     Generate the external representation of the regions
        # """
        # crime_attention = attention.unsqueeze(3).repeat(1, 1, 1, n_feature)
        # final_attention = dot_attention * crime_attention
        # ext_rep = torch.matmul(final_attention, v)  # shape = (B, N, N, dv)
        # ext_rep = ext_rep.sum(dim=2)  # shape = (B, N, N, dv)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
