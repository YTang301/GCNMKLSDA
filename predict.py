import numpy as np
import time
from model.train_model import GCNMKLSDA_main
from Input_snoRNA import Input_snoRNA
from similarity_utility import gaussiansimilarity
from openpyxl import Workbook
interaction, SDA, dis_sim, sno_sim,diseaseName,snoRNA = Input_snoRNA()
SDA[:, [0, 1]] = SDA[:, [1, 0]]

def integrate_kernels(fm, gm):
    integrated = np.where(fm != 0, (fm + gm) / 2, gm)
    return integrated

def case_study():
    nd, ns = interaction.shape
    kd, ks = gaussiansimilarity(interaction,nd,ns)

    integrated_disease_kernel = integrate_kernels(dis_sim, kd) # Integrate similarities using average
    integrated_snoRNA_kernel = integrate_kernels(sno_sim, ks)

    start = time.time()
    prediction_score_matrix = GCNMKLSDA_main(interaction, integrated_disease_kernel, integrated_snoRNA_kernel)
    end = time.time()
    print('\t', 'The mainFunc running time = ', str(format(end-start, '.4f')), end='')
    sorted_predictions = []
    for i in range(nd):
        for j in range(ns):
            ListNew = [prediction_score_matrix[i][j], i, j, interaction[i][j]]
            sorted_predictions.append(ListNew)
    sorted_predictions.sort(reverse=True)
    wb = Workbook()
    ws = wb.active # Get the default active worksheet
    ws.append(["Disease", "snoRNA gene ID","snoRNA gene symbol", "Score", "Label"]) # Write the header row
    for i in range(len(sorted_predictions)): # Write data rows
               ws.append([diseaseName[sorted_predictions[i][1]][0], snoRNA[sorted_predictions[i][2]][0] ,snoRNA[sorted_predictions[i][2]][1],sorted_predictions[i][0], sorted_predictions[i][3]])
    wb.save("case_study_result.xlsx") # Replace with your own path
    print('\n\t', 'All procedures have been completed successfully.', '\n')
    return 0