import numpy as np
import pandas as pd
import os

def Input_snoRNA():
    r0 = './dataset/'
    disease_sim = pd.read_csv(os.path.join(r0, 'disease_sim_graph_filtered.csv'), header=None).values
    snoRNA_sim = pd.read_csv(os.path.join(r0, 'snoRNA_4mer_similarity.csv'), header=None).values
    interaction = pd.read_csv(os.path.join(r0, 'relationship_matrix_filtered.csv'), header=None).values  # Rows are diseases, columns are snoRNAs
    disease = pd.read_excel(os.path.join(r0, 'disease_name.xlsx'), header=None).values.tolist() 
    snoRNA =  pd.read_excel(os.path.join(r0, 'snoRNA_name.xlsx'), header=None).values.tolist()
    rows, columns = interaction.shape
    SDA = np.zeros((sum(sum(interaction)), 2)) # Initialize SDA, interaction is a 2D matrix, nested summing sum(sum(interaction)), get the number of associations, 2 indicates two columns
    n = 0
    for i in range(0, rows): # Note that Python indexing starts from 0!!!
        for j in range(0, columns):
            if interaction[i, j] == 1: # Check if an association exists, then store the coordinates of elements with value 1 in the interaction matrix
                   SDA[n, 0] = i + 1 # Pay attention to whether to add 1, it must be consistent later; starting from 0, +1 means coordinates starting from 1
                   SDA[n, 1] = j + 1
                   n = n + 1
    SDA = SDA.astype(np.int64)
    print('\n')
    print('************************************************', '\n')
    print('\t', 'SDAs of All_Input() is done!', '\n')
    print('************************************************')
    diseaseName = disease
    return interaction, SDA, disease_sim, snoRNA_sim, diseaseName, snoRNA
    # Return data format:
    # interaction 27×220, SDA 459×2 (left large, right small), Kd 27 27 1, Ks 220 220 1, disease_sim 27×27, snoRNA_sim 220×220, diseaseName 27 list, snoRNA 220 list