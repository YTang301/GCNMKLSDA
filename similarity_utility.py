import numpy as np
import copy

def gaussiansimilarity(interaction, nd, ns):
    gamad = nd/((np.linalg.norm(interaction))**2)
    interaction_copy = copy.deepcopy(interaction)
    kd = np.zeros([nd, nd])
    disease_similarity_matrix = np.dot(interaction_copy, interaction_copy.T)
    for i in range(nd):
        for j in range(nd):
            kd[i, j] = np.exp(-gamad*(disease_similarity_matrix[i, i]+disease_similarity_matrix[j, j]-2*disease_similarity_matrix[i, j]))
    gamam = ns/((np.linalg.norm(interaction))**2) # calculate gamam for Gaussian kernel calculation + 10**(-20)
    ks = np.zeros([ns, ns]) # calculate Gaussian kernel for the similarity between snoRNA: ks
    sno_similarity_matrix = np.dot(interaction_copy.T, interaction_copy)
    for i in range(ns):
        for j in range(ns):
            ks[i, j] = np.exp(-gamam*(sno_similarity_matrix[i, i]+sno_similarity_matrix[j, j]-2*sno_similarity_matrix[i, j]))
    return kd, ks