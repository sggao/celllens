# 🎯 CellLENS

[![Python Versions](https://img.shields.io/pypi/pyversions/cellsnap.svg)](https://pypi.org/project/cellsnap)
[![PyPI Version](https://img.shields.io/pypi/v/cellsnap.svg)](https://pypi.org/project/cellsnap)
[![GitHub Issues](https://img.shields.io/github/issues/sggao/cellsnap.svg)](https://github.com/sggao/cellsnap/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sggao/cellsnap/blob/master/tutorials/CellSNAP_codex_murine.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="https://github.com/sggao/CellLENS/blob/main/media/figure1_v4.png" width="80%">
</p>

## 📌 Overview
**Cell Local Environment Neighborhood Scan (CellLENS)** is a computational method that integrates cross-domain information from tissue samples to learn a single-cell representation embedding. By analyzing spatial proteomic and spatial transcriptomic datasets across different tissue types and disease settings, CellLENS identifies biologically relevant cell populations that were previously challenging to detect due to lost tissue morphological information.

For more details, see our [preprint](https://www.biorxiv.org/content/10.1101/2024.05.12.593710v1).

> ⚠️ **Active Development:** This repository is under active development. The current version is for **reviewing and early access testing**. A full installation guide and tutorial will be available soon.

---

## 🚀 Installation
CellLENS is hosted on `pypi` and can be installed via `pip`. We recommend working within a virtual environment. The package requires `CUDA` support as it uses `PyTorch`.

```bash
conda create -n celllens python=3.9   # Create a new environment
conda activate celllens               # Activate environment
pip install celllens==0.1.0           # Install CellLENS
```
After installation, import the module as:
```python
import celllens
```
For usage details, check out our **[tutorials](#-tutorials)**.

---

## 📖 Tutorials

📌 **[Tutorial I](https://github.com/sggao/celllens/blob/main/tutorials/Tutorial_LITE_CellLENS.ipynb)** - Simplified usage with feature expression & cell locations.

📌 **[Tutorial II](https://github.com/sggao/celllens/blob/main/tutorials/Tutorial_Full_CellLENS.ipynb)** - Full version using feature expression, cell locations, & tissue images.

📌 **[Tutorial III](https://github.com/sggao/celllens/blob/main/tutorials/Optionl_Tutorial_Full_CellLENS-ViT.ipynb)** - Advanced usage with **ViT-based** image feature learning.

---

## ⚙️ Key Parameters & Recommendations

| **Parameter**         | **Function**         | **Description**  | **Default** | **Recommendation** |
|----------------------|---------------------|------------------|------------|--------------------|
| `nbhd_composition`  | `SNAP_Dataset()`     | Nearest neighbors used for learning local cellular patterns. | 20 | Tune from **10-50** based on local structure. |
| `feature_neighbor`  | `SNAP_Dataset()`     | Feature similarity graph nearest neighbors for GNN training. | 15 | Usually no need to change. |
| `spatial_neighbor`  | `SNAP_Dataset()`     | Spatial similarity graph nearest neighbors for GNN training. | 15 | Usually no need to change. |
| `pca_components`    | `.initialize()`      | PCA components for reduced feature expression input. | 25 | User-defined, similar to scRNA-seq PCA selection. |
| `celltype`         | `.initialize()`      | Initial cell type labels. `feature_labels` uses Leiden clustering. | `feature_labels` | Use if predefined labels are unavailable. |
| `cluster_res`       | `.initialize()`      | Leiden clustering resolution. | 0.5 | Adjust if dataset is too large/small. |
| `n_clusters`        | `.initialize()`      | Number of clusters for Leiden clustering. | None | Specify **8-15** clusters for stability. |
| `size`             | `.prepare_images()`  | Pixel size for CNN-based tissue morphology analysis. | 512 | 50-200 μm works well. |
| `cnn_latent_dim`   | `CellSNAP()`         | Latent dimension from CNN image features. | 128 | No need to change. |
| `gnn_latent_dim`   | `CellSNAP()`         | Latent dimension from GNN feature fusion. | 32 | No need to change. |
| `round`            | `.get_snap_embedding()` | Training rounds for the CellLENS duo-GNN model. | 5 | No need to change. |
| `k`               | `.get_snap_embedding()` | SVD embedding dimension after duo-GNN training. | 32 | No need to change. |

---

## 📢 Citation
If you find **CellLENS** useful, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2024.05.12.593710v1).
```
@article{yourcitation2024,
  author    = {Your Name et al.},
  title     = {CellLENS: A Spatial Multi-Omics Representation Learning Method},
  journal   = {bioRxiv},
  year      = {2024},
  doi       = {10.1101/2024.05.12.593710v1}
}
```

---

## 📬 Contact & Contributions
We welcome contributions and feedback! Please open an [issue](https://github.com/sggao/cellsnap/issues) or submit a pull request.

---
