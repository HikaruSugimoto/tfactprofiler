# TFActProfiler

**TFActProfiler** provides tools to infer transcription factor (TF) activities from gene expression data, assess their reliability, and simulate perturbation effects with or without additional model training.   

A web application for visualizing the TF–mRNA database and estimating TF activity from transcriptomic data (https://tfestimatetest.streamlit.app/) 

## Features
1. **TF activity inference with reliability estimation**  
   - `estimate_reliability`: Computes activity scores for transcription factors and estimates their reliability 
   
2. **Perturbation simulation without additional training**  
   - `perturbation_predict`: Predicts the effects of TF perturbations (e.g., knockout) directly from prior TF–target interaction networks and observed expression data.  
   - No extra model fitting required.
   
3. **Perturbation simulation with additional training**  
   - `train_W`, `predict_withW`: Learns gene expression changes to predict TF perturbations.


## Installation
```bash
pip install tfactprofiler
```

## Quick usage
Usage examples are provided as Jupyter notebooks inside each **example** folder:

- `example/TF_activity_inference.ipynb`  
  *Single-cell TF activity inference with reliability estimation.*
  
- `example/Perturbation_without_training.ipynb`  
  *Perturbation simulation without additional model training.*
  
- `example/Perturbation_with_training.ipynb`  
  *Perturbation simulation with model training and cross-validation.*

## Methods
TFActProfiler was constructed by first assembling candidate TF–mRNA regulatory pairs from multiple prior resources, including CellOracle-base networks, ChIP-Atlas, and CollecTRI. These candidate interactions were then integrated with large-scale bulk and single-cell RNA-seq datasets, and target-gene expression was modeled as a function of TF expression using regularized regression in a cluster-wise manner to estimate signed regulatory coefficients. During coefficient estimation, putative self-activating TF→TF relationships were temporarily excluded from the regression step to avoid bias in the inferred regulatory effects. After regression and coefficient aggregation, these self-regulatory relationships were re-integrated into the final TF–mRNA interaction resource.

It should be noted that the perturbation-prediction component of TFActProfiler is unlikely to represent the optimal approach, given the rapid emergence of numerous methods for predicting transcriptional responses to perturbations in recent years. Rather, this analysis is included primarily as a qualitative visualization to demonstrate that the TF–mRNA regulatory relationships incorporated in TFActProfiler contain biologically plausible interactions to a certain extent. Therefore, perturbation predictions from TFActProfiler should be interpreted in conjunction with results obtained using other complementary methods. A major intended application of TFActProfiler is the inference of upstream transcription factor activity from lists of differentially expressed mRNAs obtained, for example, by comparing disease and healthy conditions. Combining TFActProfiler with established TF activity inference frameworks and regulatory resources, such as CollecTRI and ChEA3, may enable more robust, accurate, and comprehensive estimation of transcription factor activity.

## Citation
Sugimoto, H., Tsuyuzaki, K., Zou, Z., Oki, S., Ohta, T., & Kawakami, E. (2026). A transcription factor regulatory atlas for activity inference and perturbation prediction. Nucleic Acids Research, 54(17), gkag897.