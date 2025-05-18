from model import GCNMKLSDA
import numpy as np
from model.utils import constructHNet, constructNet, get_edge_index, Sizes
import torch as t
from torch import optim
from model.loss import Myloss

def train(model, train_data, optimizer, sizes):
    model.train()
    regression_crit = Myloss()

    def train_epoch(): # Function to train one epoch
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(train_data['M'], score, model.l_d, model.l_s, model.alpha1, # Compute the loss
                               model.alpha2, sizes)
        
        # Eq.(14)
        model.alpha1 = t.mm( # Update alpha1 and alpha2
            t.mm((t.mm(model.N_d, model.N_d) + model.lambda1 * model.l_d).inverse(), model.N_d),
            2 * train_data['M'] - t.mm(model.alpha2.T, model.N_s.T)).detach()
        # Eq.(15)
        model.alpha2 = t.mm(t.mm((t.mm(model.N_s, model.N_s) + model.lambda2 * model.l_s).inverse(), model.N_s),
                            2 * train_data['M'].T - t.mm(model.alpha1.T, model.N_d.T)).detach()
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        return loss
    for epoch in range(1, sizes.epoch + 1): # Loop over epochs and train
        train_reg_loss = train_epoch()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass

def PredictScore(train_dis_sno_matrix, dis_matrix, sno_matrix, seed, sizes):
    np.random.seed(seed)
    train_data = {}
    train_data['M'] = t.DoubleTensor(train_dis_sno_matrix)

    Heter_adj = constructHNet(train_dis_sno_matrix, dis_matrix, sno_matrix) # Construct heterogenous network
    Heter_adj = t.FloatTensor(Heter_adj)
    Heter_adj_edge_index = get_edge_index(Heter_adj)
    train_data['X'] = {'data': Heter_adj, 'edge_index': Heter_adj_edge_index}
    X = constructNet(train_dis_sno_matrix) # Construct network features
    X = t.FloatTensor(X)
    train_data['feature'] = X
    model = GCNMKLSDA.Model(sizes, dis_matrix, sno_matrix) # Initialize model
    print(model)
    for parameters in model.parameters():
        print(parameters, ':', parameters.size())
    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate) # Set optimizer
    train(model, train_data, optimizer, sizes) # Train the model
    return model(train_data) # Return the predicted scores

def GCNMKLSDA_main(dis_sno_matrix, dis_sim, sno_sim):
    sizes = Sizes(dis_sim.shape[0], sno_sim.shape[0])
    dis_sno_res = PredictScore(dis_sno_matrix, dis_sim, sno_sim, sizes.seed, sizes) # Predict scores
    dis_sno_res = dis_sno_res.detach().numpy() # Convert the predicted results to a numpy array and return
    return dis_sno_res