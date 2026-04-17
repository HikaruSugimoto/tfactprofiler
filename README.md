# TFActProfiler

**TFActProfiler** provides tools to infer transcription factor (TF) activities from gene expression data, assess their reliability, and simulate perturbation effects with or without additional model training.

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
