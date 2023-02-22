import argparse as Ap
import glob
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from chicago_feature_representation import *
from layers import *
from model import *
from similarity_measurement import *

seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)


chicago_region = 73
current_crime_category = 0
k = 16
lr = 0.004# initial learning rate 0.004
epoch_total = 321
# main model

argp = Ap.ArgumentParser()

argp.add_argument("--ts", default=20, type=int, help="Number of time-steps")
argp.add_argument("--in_hgat", default=1, type=int, help="input dimension of hgat layers")
argp.add_argument("--in_fgat", default=12, type=int, help="input dimension of fgat layers:num of features")
argp.add_argument("--out_gat", default=8, type=int, help="output dimension of both hgat and fgat layers")
argp.add_argument("--att_dot", default=40,type=int, help="(dot-product)attention dimension of fgat")
argp.add_argument("--nhid_rnn",default=40,type=int, help="hidden dimension of rnn") # previous 40
argp.add_argument("--nlayer_rnn",default=1,type=int, help="number of rnn layers") # previous 1
argp.add_argument("--att_rnn",default=30,type=int, help="(location)attention dimension of temporal module")

d = argp.parse_args(sys.argv[1:])





# location = "chicago"
target_region = chicago_region
target_cat = current_crime_category
ts = d.ts
in_hgat = d.in_hgat
in_fgat = d.in_fgat
out_gat = d.out_gat
att_dot = d.att_dot
nhid_rnn = d.nhid_rnn
nlayer_rnn = d.nlayer_rnn
att_rnn = d.att_rnn
batch_size = 42
location = "chicago"

# gen_gat_adj_file(target_region, location)   # generate the adj_matrix file for GAT layers
loaded_data = torch.from_numpy(np.loadtxt("chicago_data/" + location + "/com_crime/r_" + str(chicago_region) + ".txt", dtype=int)).T
loaded_data = loaded_data[:, current_crime_category:current_crime_category+1]

tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=int))
x, Y, x_daily, x_weekly = create_inout_sequences(loaded_data)

scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
x_daily = torch.from_numpy(scale.fit_transform(x_daily))
x_weekly = torch.from_numpy(scale.fit_transform(x_weekly))
Y = torch.from_numpy(scale.fit_transform(Y))


# load data
train_x, train_x_daily, train_x_weekly, train_y, test_x, test_x_daily, test_x_weekly, test_y = load_self_crime(x, x_daily,x_weekly, Y)
with open('chicago_data/chicago/formatted_taxi_data_chicago/inflows_train.pkl', 'rb') as f:
    inflows_train = pickle.load(f)
with open('chicago_data/chicago/formatted_taxi_data_chicago/inflows_test.pkl', 'rb') as f:
    inflows_test = pickle.load(f)
with open('chicago_data/chicago/formatted_taxi_data_chicago/outflows_train.pkl', 'rb') as f:
    outflows_train = pickle.load(f)
with open('chicago_data/chicago/formatted_taxi_data_chicago/outflows_test.pkl', 'rb') as f:
    outflows_test = pickle.load(f)

# 6. (Loss Function): MSE on NYC crime + loss of Auto-encoder in (1)

model = ModifiedAIST(chicago_region, in_hgat, in_fgat, out_gat, att_dot, nhid_rnn, nlayer_rnn, att_rnn, ts, batch_size)

n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:", n)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
criterion = nn.L1Loss()
epochs = 300
best = epochs + 1
best_epoch = 0
t_total = time.time()
loss_values = []
bad_counter = 0
patience = 100

train_batch = train_x.shape[0]
test_batch = test_x.shape[0]


print("Chicago region : ", chicago_region)
print("Crime type : ", current_crime_category)


crime_map ={}
crime_map[0] = 1
crime_map[1] = 4
crime_map[2] = 5
crime_map[3] = 6
crime_map[4] = 7


for epoch in range(epochs):
    
    print("Chicago epoch : ", epoch)
  
    for i in range(train_batch): # 33

        print("Train batch: ", i)
        model.train()
        optimizer.zero_grad()
        output, y_true = model(chicago_region , i, batch_size, k, crime_map[current_crime_category], loaded_data, 0, current_crime_category, inflows_train, outflows_train)
        loss_train = criterion(output, y_true)
        #print("Loss : ", loss_train)
        loss_train.backward()
        optimizer.step()
        loss_values.append(loss_train)

        with open("best_epoch_models_chicago/" + str(chicago_region) + "/" + str(current_crime_category) + "/{}_{}_{}.pkl".format(chicago_region, current_crime_category, epoch*train_batch + i + 1), "wb") as f:
            pickle.dump(model.state_dict(), f)
        
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch*train_batch + i + 1
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('best_epoch_models_chicago/'+str(chicago_region)+'/'+str(current_crime_category) + '/*.pkl')
        for file in files:
            remove_file = file
            file = file.split("/")
            file = file[len(file) - 1]
            file = file.split("\\")[1]
        
            if file.startswith('{}_{}'.format(chicago_region, current_crime_category)):
                epoch_nb = file.split('.')[0]
                epoch_nb = int(epoch_nb.split('_')[2])
                if epoch_nb < best_epoch:
                    os.remove(remove_file)
                    
    files = glob.glob('best_epoch_models_chicago/'+str(chicago_region)+'/'+str(current_crime_category) + '/*.pkl')
    for file in files:
        remove_file = file
        file = file.split("/")
        file = file[len(file) - 1]
        file = file.split("\\")[1]
        if file.startswith('{}_{}'.format(chicago_region, current_crime_category)):
            epoch_nb = file.split('.')[0]
            epoch_nb = int(epoch_nb.split('_')[2])
            if epoch_nb > best_epoch:
                os.remove(remove_file)
               
    if epoch*train_batch + i + 1 >= epoch_total:
        break
    
print("Optimization Finished for Chicago!")
print('Loading {}th epoch'.format(best_epoch))


with open("best_epoch_models_chicago/"+str(chicago_region)+"/"+str(current_crime_category) + "/{}_{}_{}.pkl".format(chicago_region, current_crime_category, best_epoch), "rb") as f:
    params = pickle.load(f)
model.load_state_dict(params)




losses = [(float) (i.detach()) for i in loss_values]
plt.plot(np.array(losses), 'r')
plt.savefig("Training Losses/" + str(chicago_region) + "_" + str(current_crime_category) + "_" + str(k) + "_MAE_loss_weighted_new.jpg")



# # test
# inflow_test, outflow_test = handle_taxi_data(1, train_batch, test_batch)
loss = 0
for i in range(test_batch):
    model.eval()
    output_test, y_similar_regions_chicago = model(chicago_region , i, batch_size, k, crime_map[current_crime_category], loaded_data, 1, current_crime_category, inflows_test, outflows_test)
    y_test = Variable(test_y[i]).float()
    y_test = y_test.view(-1, 1)
    y_test = torch.from_numpy(scale.inverse_transform(y_test))
    print("True", y_test)
    output_test = torch.from_numpy(scale.inverse_transform(output_test.detach()))
    print("Prediction: ", output_test)
    loss_test = criterion(output_test, y_test)
    
    loss += loss_test.data.item()
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()))


print(chicago_region, " ", target_cat, " ", loss/i)
