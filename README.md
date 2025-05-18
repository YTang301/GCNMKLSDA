
# GCNMKLSDA

This repository provides the implementation of the **case study** for our manuscript:

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


2. **Replace all data paths in `.py` files**:
   - Modify path strings in:
     - `main.py`
     - `predict.py`
     - `Input_snoRNA.py`


3. **Run**:
   python main.py


4. **Output**:
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


##  Citations

We acknowledge and thank the following foundational works:

> Yang H, Ding Y, Tang J, et al.
> **Inferring human microbe–drug associations via multiple kernel fusion on graph neural network**.  
> *Knowledge-Based Systems*, 2022, 238: 107888.
> [ GitHub: hhttps://github.com/guofei-tju/MKGCN](https://github.com/guofei-tju/MKGCN)

> Sun Z, Huang Q, Yang Y, et al.  
> **PSnoD: identifying potential snoRNA-disease associations based on bounded nuclear norm regularization**.  
> *Briefings in Bioinformatics*, 2022, 23(4): bbac240.
> [ GitHub: https://github.com/linDing-groups/PSnoD](https://github.com/linDing-groups/PSnoD)

 **Note**: If you use this code or cite our work, please also cite both of the above papers as we build upon their methods and datasets.

The baseline methods we used are described in the following publications or GitHub repositories.

> Liu D, Y Luo, J Zheng, et al.
> **GCNSDA: Predicting snoRNA-disease associations via graph convolutional network**.  
> *Proceedings of the IEEE BIBM*, 2021, pp. 183–188. doi: [10.1109/BIBM52615.2021.9669505]

> Momanyi B M, Y-W Zhou, B K Grace-Mercure, et al.  
> **SAGESDA: Multi-GraphSAGE networks for predicting SnoRNA-disease associations**.  
> *Current Research in Structural Biology*, 2024, 7:100122. doi: [10.1016/j.crstbi.2023.100122]
> [ https://github.com/momanyibiffon/SAGESDA](https://github.com/momanyibiffon/SAGESDA)

> Zhang W, B Liu  
> **iSnoDi-LSGT: identifying snoRNA-disease associationsbased on local similarity constraints and globaltopological constraints**.  
> *RNA*, 2022, 28(12):1558–1567. doi: [10.1261/rna.079325.122]
> Web Server: http://bliulab.net/iSnoDi-LSGT

> Hu X, P Zhang, D Liu, et al.  
> **IGCNSDA: unraveling disease-associated snoRNAs with an interpretable graph convolutional network**.  
> *Briefings in Bioinformatics*, 2024, 25(3):bbae179. doi: [10.1093/bib/bbae179]
> [https://github.com/altriavin/IGCNSDA](https://github.com/altriavin/IGCNSDA)

> Chen X, L Wang, J Qu, et al.  
> **Predicting miRNA–disease association based on inductive matrix completion**.  
> *Biomolecules*, 2022, 12(1):64. doi: [10.3390/biom12010064]
> [https://github.com/lazywolf007/IMCMDA](https://github.com/lazywolf007/IMCMDA)

> Chen X, J Yin, J Qu, et al.  
> **MDHGI: Matrix Decomposition and Heterogeneous Graph Inference for miRNA-disease association prediction**.  
> *PLOS Computational Biology*, 2018, 14(8):e1006418. doi: [10.1371/journal.pcbi.1006418]
> [https://github.com/wengelearning/MDHGI](https://github.com/wengelearning/MDHGI)



##  Contact

For any questions or suggestions:

> **Yong Tang**, Department of Medical Innovation and Research, Chinese PLA General Hospital
> **Email**: tangyong_301@163.com (corresponding author)
