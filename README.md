
# GCNMKLSDA

This repository provides the implementation of our manuscript:

> **"A Graph Convolutional Network Framework with Similarity Integration for Predicting snoRNA–Disease Associations"**  
> *(Submitted to Briefings in Bioinformatics, 2025)*

This repository enables reproducible prediction of potential snoRNA–disease associations, and outputs all results in a structured `.xlsx` file.


##  Overview

GCNMKLSDA integrates similarity fusion and graph convolutional networks (GCN) to construct an effective predictor for identifying snoRNA–disease associations.



##  How to Run

1. **Install Python dependencies**:
   numpy==1.26.4
   pandas==2.2.2
   openpyxl==3.1.2
   torch==2.3.0
   torch_geometric==2.5.3


2. **Run**:
   python main.py


3. **Output**:
   - `case_study_result.xlsx` — includes the predicted score for each `(disease, snoRNA)` pair, with headers:
     
     Disease | snoRNA gene ID | snoRNA gene symbol | Score | Label



##  Data Description

The dataset is derived from [MNDR v3.1](http://www.rna-society.org/mndr/) and includes:
- 220 snoRNAs
- 27 diseases
- 459 validated associations

Integrated similarities:
- snoRNAs: 4-mer frequency (via k-mer counting) and GIP
- Diseases: semantic similarity (MeSH DAG) and GIP

For details, refer to our manuscript’s **Section 2.1 – 2.4**.



##  Contact

For any questions or suggestions:

> **Yong Tang**, Department of Medical Innovation and Research, Chinese PLA General Hospital
> **Email**: tangyong_301@163.com (corresponding author)
