import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import MinMaxScaler


def create_inout_sequences(input_data, tw=120):
    """
    :param input_data: raw linear data of a region
    :param tw: length of a sample
    """
    forecast = 1  # Num of days to forecast in the future
    # print(input_data.shape)

    # Consecutive_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int))
    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(
            train_label.shape[0] * train_label.shape[1])
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    # Daily_temporal_data_generation
    num_samples = in_seq1.shape[0]
    time_step_daily = int(tw / 6)
    in_seq2 = torch.from_numpy(
        np.ones((num_samples, time_step_daily), dtype=np.int))
    for i in range(num_samples):
        k = 0
        for j in range(tw):
            if j % 6 == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # Weekly_temporal_data_generation
    time_step_weekly = int(tw / (6 * 7)) + 1
    in_seq3 = torch.from_numpy(
        np.ones((num_samples, time_step_weekly), dtype=np.int))
    for i in range(num_samples):
        k = 0
        for j in range(tw):
            if j % (6 * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3


def load_data_GAT():

    # build features
    # (Nodes, NodeLabel+ features + label)
    idx_features_labels = np.genfromtxt(
        "chicago_data/gat_crime.txt", dtype=np.dtype(str))
    features = sp.csr_matrix(
        idx_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features_ext
    idx_features_labels_ext = np.genfromtxt("chicago_data/gat_ext.txt",
                                            dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features_ext = sp.csr_matrix(
        idx_features_labels_ext[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features
    idx_crime_side_features_labels = np.genfromtxt("chicago_data/gat_side.txt",
                                                   dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    crime_side_features = sp.csr_matrix(
        idx_crime_side_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)

    # build graph
    num_reg = int(idx_features_labels.shape[0] / (42))
    # replaced 5
    idx = np.array(idx_features_labels[:num_reg, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("chicago_data/gat_adj.txt", dtype=np.int32)

    if edges_unordered.ndim == 1 and edges_unordered.shape[0] == 2:
        edges_unordered = edges_unordered.reshape([1, 2])

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_reg, num_reg),
                        dtype=np.float32)  # replaced 5
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    features_ext = torch.FloatTensor(np.array(features_ext.todense()))
    crime_side_features = torch.FloatTensor(
        np.array(crime_side_features.todense()))

    return adj, features, features_ext, crime_side_features


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_self_crime(x, x_daily, x_weekly, y):
    """
    :param x:
    :param x_daily:
    :param x_weekly:
    :param y:
    """
    batch_size = (42)
    time_step = 120
    train_ratio = 0.67

    train_x_size = int(x.shape[0] * train_ratio)

    train_x = x[: train_x_size, :]
    train_x_daily = x_daily[: train_x_size, :]
    train_x_weekly = x_weekly[: train_x_size, :]
    train_y = y[: train_x_size, :]

    test_x = x[train_x_size:, :]
    test_x_daily = x_daily[train_x_size:, :]
    test_x_weekly = x_weekly[train_x_size:, :]
    test_x = test_x[:test_x.shape[0] - 11, :]

    # batch size
    test_x_daily = test_x_daily[:test_x_daily.shape[0] - 11, :]
    test_x_weekly = test_x_weekly[:test_x_weekly.shape[0] - 11, :]
    test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
    test_y = test_y[:test_y.shape[0] - 11, :]

    # Divide it into batches -----> (Num of Batches, batch size, time-step features)
    train_x = train_x.view(
        int(train_x.shape[0] / batch_size), batch_size, time_step)
    train_x_daily = train_x_daily.view(
        int(train_x_daily.shape[0] / batch_size), batch_size, train_x_daily.shape[1])
    train_x_weekly = train_x_weekly.view(
        int(train_x_weekly.shape[0] / batch_size), batch_size, train_x_weekly.shape[1])
    train_y = train_y.view(int(train_y.shape[0] / batch_size), batch_size, 1)

    test_x = test_x.view(
        int(test_x.shape[0] / batch_size), batch_size, time_step)
    test_x_daily = test_x_daily.view(
        int(test_x_daily.shape[0] / batch_size), batch_size, test_x_daily.shape[1])
    test_x_weekly = test_x_weekly.view(
        int(test_x_weekly.shape[0] / batch_size), batch_size, test_x_weekly.shape[1])
    test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, 1)

    return train_x, train_x_daily, train_x_weekly, train_y, test_x, test_x_daily, test_x_weekly, test_y


def load_nei_crime(target_crime_cat, target_region, location):
    """
    :param target_crime_cat: starts from 0
    :param target_region: starts from 0
    :param location: name of the city
    :return: batch_add_train:
    :return: batch_add_test:
    """
    batch_size = (42)
    time_step = 120
    train_ratio = 0.67

    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    com = gen_neighbor_index_zero(target_region, location)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt(
            "chicago_data/" + location + "/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * train_ratio)
        train_x = x[: train_x_size, :]  # (samples, time-step)
        train_y = y[: train_x_size, :]  # (samples, time-step)
        test_x = x[train_x_size:, :]  # (samples, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11, :]  # (samples, time-step)
        test_y = y[train_x_size:, :]  # (samples, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(
            int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(
            int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_all_ext(target_region, location):
    """
    :param target_region: starts from 0
    :param location: name of the city
    :return:
    """
    batch_size = 42
    time_step = 120
    train_ratio = 0.67
    nfeature = 2  # taxi inflow + taxi outflow

    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = gen_neighbor_index_one_with_target(target_region, location)
    com = [target_region+1]

    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt(
            location + "_data/" + location + "/ext/taxi" + str(i) + ".txt", dtype=int)).T
        loaded_data1 = loaded_data[:, 0:1]
        loaded_data2 = loaded_data[:, 1:2]
        x_in, y_in, z_in, m_in = create_inout_sequences(
            loaded_data1, time_step)
        x_out, y_out, z_out, m_out = create_inout_sequences(
            loaded_data2, time_step)

        scale = MinMaxScaler(feature_range=(0, 1))
        x_in = x_in.unsqueeze(2).double()
        x_out = x_out.unsqueeze(2).double()
        x = torch.cat([x_in, x_out], dim=2)
        # print(x.shape) # 2069, 120, 2

        # Divide into train_test chicago_data
        train_x_size = int(x.shape[0] * train_ratio)
        train_x = x[: train_x_size, :, :]
        test_x = x[train_x_size:, :, :]
        test_x = test_x[:test_x.shape[0] - 11, :, :]

        train_x = train_x.view(
            int(train_x.shape[0] / batch_size), batch_size, time_step, nfeature)
        test_x = test_x.view(
            int(test_x.shape[0] / batch_size), batch_size, time_step, nfeature)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    # print(len(batch_add_train))
    # print(batch_add_train[0].shape)
    # print(batch_add_train[0][0].shape)
    return batch_add_train, batch_add_test


def load_all_parent_crime(target_crime_cat, target_region, location):
    """

    :param target_crime_cat: starts with 0
    :param location: name of the city
    :param target_region: starts with 0
    :return:
    """
    batch_size = 42
    time_step = 120
    train_ratio = 0.67

    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [6, 23, 27, 31, 7]  # starts with 0
    com = gen_neighbor_index_zero_with_target(target_region, location)
    side = gen_com_side_adj_matrix(com, location)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(len(com)):
        loaded_data = torch.from_numpy(np.loadtxt(
            "chicago_data/" + location + "/side_crime/s_" + str(side[i]) + ".txt", dtype=np.int)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        tensor_ones = torch.from_numpy(
            np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
        loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * train_ratio)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11,
                        :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(
            int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(
            int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    return batch_add_train, batch_add_test


def gen_com_adj_matrix(target_region, location):

    adj_matrix = np.zeros((77, 77), dtype=np.int)
    edges_unordered = np.genfromtxt(
        "chicago_data/" + location + "/com_adjacency.txt", dtype=np.int32)
    for i in range(edges_unordered.shape[0]):
        src = edges_unordered[i][0] - 1
        dst = edges_unordered[i][1] - 1
        adj_matrix[src][dst] = 1
        adj_matrix[src][dst] = 1
    np.savetxt("chicago_data/" + location +
               "/com_adj_matrix.txt", adj_matrix, fmt="%d")
    return


def gen_com_side_adj_matrix(regions, location):
    """
    :param regions: a list of regions starting from 0
    :param location: name of the city
    :return: sides: a list of sides which are mapped (side, com) starts with 0
    """
    idx = np.loadtxt("chicago_data/" + location +
                     "/side_com_adj.txt", dtype=np.int)
    idx_map = {j: i for i, j in iter(idx)}
    side = [idx_map.get(x + 1) % 101 for x in regions]  # As it starts with 0
    return side


def gen_neighbor_index_zero(target_region, location):
    """
    :param target_region: starts from 0
    :param location: name of the city
    :return: indices of neighbors of target region (starts from 0)
    """
    adj_matrix = np.loadtxt("chicago_data/" + location + "/com_adj_matrix.txt")
    adj_matrix = adj_matrix[target_region]
    neighbors = []
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[i] == 1:
            neighbors.append(i)
    return neighbors


def gen_neighbor_index_zero_with_target(target_region, location):
    """
    :param target_region: starts from 0
    :param location: name of the city
    :return: indices of neighbors of target region (starts from 0)
    """
    neighbors = gen_neighbor_index_zero(target_region, location)
    neighbors.append(target_region)
    return neighbors


def gen_neighbor_index_one_with_target(target_region, location):
    """
    :param target_region: starts from 0
    :param location: name of the city
    :return: indices of neighbors of target region (starts from 0)
    """
    neighbors = gen_neighbor_index_zero(target_region, location)
    neighbors.append(target_region)
    neighbors = [x + 1 for x in neighbors]
    return neighbors


def gen_gat_adj_file(target_region, location):
    """
    :param target_region: starts from 0
    :param location: name of the city
    :return:
    """
    neighbors = gen_neighbor_index_zero(target_region, location)
    adj_target = torch.zeros(len(neighbors), 2)
    for i in range(len(neighbors)):
        adj_target[i][0] = target_region
        adj_target[i][1] = neighbors[i]
    np.savetxt("chicago_data/gat_adj.txt", adj_target, fmt="%d")
    return


def crime_representation(train_or_test, similar_regions, attention_values, current_crime_category, similar_region_number, batchNo):

    if train_or_test == 0:
        h_gat_representation = []
        for m in similar_regions:
            file = torch.load('h_gat_representation/' + str(m) +
                              '/' + str(current_crime_category) + '/h_gat.pkl')
            h_gat_representation.append(file)

        # # now h_gat_representation has k h_gats and we have similarity score can be found using attention_values and similar_regions

        crime_representation = 0.0
        for i in range(similar_region_number):
            h_gat_representation_for_this_precinct = torch.cat(
                h_gat_representation[i][batchNo]).reshape((42, 20, 8))
            crime_representation += ((attention_values[similar_regions[i]]) * (
                h_gat_representation_for_this_precinct))
            # crime_representation += (h_gat_representation_for_this_precinct)

        # # 5. (Input for LSTM): Concatenate Crime representation from 4 and Feature representation from 1

    elif train_or_test == 1:
        h_gat_test_representation = []

        for m in similar_regions:
            file = torch.load('h_gat_test_representation/' + str(m) +
                              '/' + str(current_crime_category) + '/h_gat_test.pkl')
            h_gat_test_representation.append(file)

        crime_representation = 0.0
        for i in range(similar_region_number):
            h_gat_test_representation_for_this_precinct = torch.cat(
                h_gat_test_representation[i][batchNo]).reshape((42, 20, 8))
            crime_representation += (h_gat_test_representation_for_this_precinct *
                                     (attention_values[similar_regions[i]]))
            # crime_representation += h_gat_test_representation_for_this_precinct

    return crime_representation


def similar_region_crimes(loaded_data, similar_regions, number_of_similar_regions, target_cat):

    loaded_data_tl = torch.zeros(loaded_data.shape)
    for i in range(number_of_similar_regions):
        loaded_data_transfer_learning_all_crimes = torch.from_numpy(np.loadtxt(
            "chicago_data/chicago/com_crime/r_" + str(similar_regions[i]) + ".txt", dtype=int)).T
        loaded_data_tl += loaded_data_transfer_learning_all_crimes[:,
                                                                   target_cat:target_cat+1]

    tensor_ones_tl = torch.from_numpy(
        np.ones((loaded_data_tl.size(0), loaded_data_tl.size(1)), dtype=int))
    x_tl, y_tl, x_daily_tl, x_weekly_tl = create_inout_sequences(
        loaded_data_tl)

    scale = MinMaxScaler(feature_range=(-1, 1))
    x_tl = torch.from_numpy(scale.fit_transform(x_tl))
    x_daily_tl = torch.from_numpy(scale.fit_transform(x_daily_tl))
    x_weekly_tl = torch.from_numpy(scale.fit_transform(x_weekly_tl))
    y_tl = torch.from_numpy(scale.fit_transform(y_tl))
    train_x_tl, train_x_daily_tl, train_x_weekly_tl, train_y_tl, test_x_tl, test_x_daily_tl, test_x_weekly_tl, test_y_tl = load_self_crime(
        x_tl, x_daily_tl, x_weekly_tl, y_tl)

    return train_y_tl, test_y_tl


# def handle_taxi_data(train_or_test, train_batch, test_batch):

#     inflows = []
#     outflows = []

#     for region in range(77):

#         train_ext, test_ext = load_all_ext(region, "chicago")

#         if train_or_test == 0:
#             taxi_data = train_ext
#             number_of_batches = train_batch
#         else:
#             taxi_data = test_ext
#             number_of_batches = test_batch

#         inflow_for_this_region = []
#         outflow_for_this_region = []

#         for i in range(number_of_batches):
#             x_ext = torch.empty((len(taxi_data[i]),taxi_data[i][0].shape[0],taxi_data[i][0].shape[1], taxi_data[i][0].shape[2]), dtype=torch.float32)
#             for j in range(len(taxi_data[i])):
#                 x_ext[j] = Variable(taxi_data[i][j]).float()
#             x_ext = torch.swapaxes(x_ext, 0, 3)
#             x_ext = torch.swapaxes(x_ext, 1, 2)
#             inflow = x_ext[0]
#             outflow = x_ext[1]
#             taxi_flow_new_shape_1 = 10
#             taxi_flow_new_shape_0 = int((inflow.shape[0] * inflow.shape[1] * inflow.shape[2]) / 10)
#             inflow = torch.reshape(inflow, (taxi_flow_new_shape_0, taxi_flow_new_shape_1))
#             outflow = torch.reshape(outflow, (taxi_flow_new_shape_0, taxi_flow_new_shape_1))

#             inflow_for_this_region.append(inflow)
#             outflow_for_this_region.append(outflow)

#         inflows.append(inflow_for_this_region)
#         outflows.append(outflow_for_this_region)

#     return inflows, outflows
