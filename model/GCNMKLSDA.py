import torch as t
from torch import nn
from torch_geometric.nn import conv
from model.utils import *

# This is the Model class for the GCNMKLSDA model
# It inherits from nn.Module and defines the layers and forward pass for the model
class Model(nn.Module):
    def __init__(self, sizes, dis_sim, sno_sim):
        super(Model, self).__init__()
        self.dis_size = sizes.dis_size
        self.sno_size = sizes.sno_size
        self.F1 = sizes.F1
        self.F2 = sizes.F2
        self.F3 = sizes.F3
        self.seed = sizes.seed
        self.h1_gamma = sizes.h1_gamma
        self.h2_gamma = sizes.h2_gamma
        self.h3_gamma = sizes.h3_gamma

        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.kernel_len = 4
        self.dis_ps = t.ones(self.kernel_len) / self.kernel_len
        self.sno_ps = t.ones(self.kernel_len) / self.kernel_len

        self.dis_sim = t.DoubleTensor(dis_sim)
        self.sno_sim = t.DoubleTensor(sno_sim)

        self.gcn_1 = conv.GCNConv(self.dis_size + self.sno_size, self.F1) # Define the GCN layers
        self.gcn_2 = conv.GCNConv(self.F1, self.F2)
        self.gcn_3 = conv.GCNConv(self.F2, self.F3)

        self.alpha1 = t.randn(self.dis_size, self.sno_size).double() # Initialize alpha1 and alpha2 matrices
        self.alpha2 = t.randn(self.sno_size, self.dis_size).double()

        self.l_d = [] # Placeholder lists for disease and snoRNA kernels
        self.l_s = []
        
        self.N_d = [] # Placeholder for disease and snoRNA kernels
        self.N_s = []

    def forward(self, input):
        x = input['feature']
        adj = input['X']
        dis_kernels = []
        sno_kernels = []
        
        # Eq.(5)
        H1 = t.relu(self.gcn_1(x, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]])) # Pass through the first GCN layer
        # Eq.(7)(8)
        dis_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.dis_size].clone(), 0, self.h1_gamma, True).double()))
        sno_kernels.append(t.DoubleTensor(getGipKernel(H1[self.dis_size:].clone(), 0, self.h1_gamma, True).double()))

        H2 = t.relu(self.gcn_2(H1, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]])) # Pass through the second GCN layer
        dis_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.dis_size].clone(), 0, self.h2_gamma, True).double()))
        sno_kernels.append(t.DoubleTensor(getGipKernel(H2[self.dis_size:].clone(), 0, self.h2_gamma, True).double()))
        
        H3 = t.relu(self.gcn_3(H2, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]])) # Pass through the third GCN layer
        dis_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.dis_size].clone(), 0, self.h3_gamma, True).double()))
        sno_kernels.append(t.DoubleTensor(getGipKernel(H3[self.dis_size:].clone(), 0, self.h3_gamma, True).double()))

        dis_kernels.append(self.dis_sim) # Add disease and snoRNA similarity to kernels
        sno_kernels.append(self.sno_sim)

        # Eq.(9)(10)
        N_d = sum([self.dis_ps[i] * dis_kernels[i] for i in range(len(self.dis_ps))]) # Calculate the final disease and snoRNA kernels using weighted sums
        self.N_d = normalized_kernel(N_d)
        N_s = sum([self.sno_ps[i] * sno_kernels[i] for i in range(len(self.sno_ps))])
        self.N_s = normalized_kernel(N_s)

        # Eq.(12)(13)
        self.l_d = laplacian(N_d) # Compute the Laplacian for disease and snoRNA kernels
        self.l_s = laplacian(N_s)
        
        # Eq.(16)
        out1 = t.mm(self.N_d, self.alpha1) # Calculate the final output using the Laplacians and alpha matrices
        out2 = t.mm(self.N_s, self.alpha2)

        out = (out1 + out2.T) / 2

        return out