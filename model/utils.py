import numpy as np
import torch as t

def constructNet(dis_sno_matrix):
    dis_matrix = np.matrix( # Create empty matrices for diseases and snoRNAs
        np.zeros((dis_sno_matrix.shape[0], dis_sno_matrix.shape[0]), dtype=np.int8))
    sno_matrix = np.matrix(
        np.zeros((dis_sno_matrix.shape[1], dis_sno_matrix.shape[1]), dtype=np.int8))    
    mat1 = np.hstack((dis_matrix, dis_sno_matrix)) # Combine disease-snoRNA matrix with the identity matrices
    mat2 = np.hstack((dis_sno_matrix.T, sno_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(dis_sno_matrix, dis_matrix, sno_matrix):
    mat1 = np.hstack((dis_matrix, dis_sno_matrix)) # Combine matrices to form the heterogeneous network
    mat2 = np.hstack((dis_sno_matrix.T, sno_matrix))
    return np.vstack((mat1, mat2))

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]): # Loop through the matrix to get the indices of non-zero elements
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)

def laplacian(kernel):
    degree_vector_D = sum(kernel) # Compute the degree matrix and the Laplacian
    degree_matrix_D = t.diag(degree_vector_D)
    unnormalized_Laplacian_L = degree_matrix_D - kernel
    D_inv_sqrt = degree_matrix_D.rsqrt()
    D_inv_sqrt = t.where(t.isinf(D_inv_sqrt), t.full_like(D_inv_sqrt, 0), D_inv_sqrt)
    normalized_Laplacian_L = t.mm(D_inv_sqrt, unnormalized_Laplacian_L)
    normalized_Laplacian_L = t.mm(normalized_Laplacian_L, D_inv_sqrt)
    return normalized_Laplacian_L

def normalized_embedding(embeddings):
    embed_min = t.amin(embeddings, 1, True)
    embed_max = t.amax(embeddings, 1, True)
    ne = (embeddings - embed_min)/(embed_max - embed_min + 1e-6)
    return ne.cpu()

def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))  # Normalize kernel by the mean of the diagonal elements
    krnl = t.exp(-kernelToDistance(krnl) * gamma)  # Apply exponential decay based on distance
    return krnl

def kernelToDistance(k):
    kernel_diagonal = t.diag(k).T
    distance_matrix = kernel_diagonal.repeat(len(k)).reshape(len(k), len(k)).T + kernel_diagonal.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return distance_matrix

def normalized_kernel(kernel_matrix):
    kernel_matrix = abs(kernel_matrix)
    flattened_kernel_values = kernel_matrix.flatten().sort()[0]
    min_nonzero_value = flattened_kernel_values[t.nonzero(flattened_kernel_values, as_tuple=False)[0]]
    kernel_matrix[t.where(kernel_matrix == 0)] = min_nonzero_value
    degree_vector = t.diag(kernel_matrix)
    degree_vector = degree_vector.sqrt()
    normalized_kernel_matrix = kernel_matrix / (degree_vector * degree_vector.T)
    return normalized_kernel_matrix

class Sizes(object):
    def __init__(self, dis_size, sno_size):
        self.dis_size = dis_size
        self.sno_size = sno_size
        self.F1 = 96
        self.F2 = 48
        self.F3 = 24
        self.epoch = 5
        self.learn_rate = 0.001
        self.seed = 1
        self.h1_gamma = 2 ** (-4)
        self.h2_gamma = 2 ** (-5)
        self.h3_gamma = 2 ** (-3)
        self.lambda1 = 2 ** (4)
        self.lambda2 = 2 ** (3)