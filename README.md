# Low_budget_MT

This repository contains experiments and scripts used for evaluating COMET-QE and Round Trip Translation Likelihood (RTTL) as active learning strategies in low-resource machine translation (MT). The project spans multiple languages: Swahili, Amharic, Kinyarwanda, and Spanish.

```bash
Low_budget_MT/
└── notebooks/
    ├── Language-specific baselines (e.g., swahili_baseline.ipynb)
    ├── COMET-QE selection scripts (e.g., swahili_COMET_QE.ipynb)
    ├── RTTL scoring and selection (e.g., swahili_RTTL.ipynb)
    ├── NRTTL experiments (e.g., Kinyarwanda_NRTTL_scoring.ipynb)
    ├── comet_scoring.ipynb
    ├── cleaning_and_viz.ipynb
    ├── data_split.ipynb
    ├── results.ipynb
    ├── my_helpers.py (utilities and shared functions)
    ├── result.png / bar_plot_with_error_bars.png (visualizations)
    └── legacy_codes.ipynb (archived/deprecated snippets)
```

## 🚀 Goal

To improve active learning in low-resource NMT by comparing:

- ✅ COMET-QE: A reference-free quality estimation metric  
- 🔁 RTTL (Round Trip Translation Likelihood): With or without noise  
- 🎯 Baselines: Random sentence selection  

## 🧪 Languages Covered

- Swahili  
- Amharic  
- Kinyarwanda  
- Spanish (used as a mid-resource reference) 

Each language has three key notebooks:

- *_baseline.ipynb: Training and evaluation on randomly selected data  
- *_COMET_QE.ipynb: Selection using COMET-QE  
- *_RTTL.ipynb: Selection using RTTL  
- (Optional) *_NRTTL.ipynb: Selection using a noisy RTTL variant  

## 📊 Results

You can find summarized plots and evaluation metrics in:

- results.ipynb  
- result.png  
- bar_plot_with_error_bars.png  

These compare BLEU scores across different selection strategies and dataset sizes.

## 🧰 Dependencies

This project uses:

- Python ≥ 3.7
- PyTorch
- JoeyNMT
- HuggingFace Transformers & Datasets
- COMET-QE (via HuggingFace or Unbabel's COMET)
- sacreBLEU
- Matplotlib, Pandas, NumPy

## 📖 How to Use

1. Preprocess and clean your dataset using cleaning_and_viz.ipynb.  
2. Split the data using data_split.ipynb.  
3. For each language, run:  
    - Baseline: *_baseline.ipynb  
    - COMET-QE selection: *_COMET_QE.ipynb  
    - RTTL selection: *_RTTL.ipynb  
4. View and compare results in results.ipynb.
