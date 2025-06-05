# 🎯 CellLENS

[![Python Versions](https://img.shields.io/pypi/pyversions/cellsnap.svg)](https://pypi.org/project/celllens)
[![PyPI Version](https://img.shields.io/pypi/v/cellsnap.svg)](https://pypi.org/project/celllens)
[![GitHub Issues](https://img.shields.io/github/issues/sggao/cellsnap.svg)](https://github.com/sggao/celllens/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sggao/celllens/blob/master/tutorials/Tutorial_LITE_CellLENS.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="https://github.com/sggao/CellLENS/blob/main/media/figure1_v4.png" width="80%">
</p>

## 📌 Overview
**Cell Local Environment Neighborhood Scan (CellLENS)** is a computational method that integrates cross-domain information from tissue samples to learn a single-cell representation embedding. By analyzing spatial proteomic and spatial transcriptomic datasets across different tissue types and disease settings, CellLENS identifies biologically relevant cell populations that were previously challenging to detect due to lost tissue morphological information.

For more details, see our [paper](https://www.nature.com/articles/s41590-025-02163-1).

> ⚠️ **Active Development:** This repository is under active development. The current version is for **reviewing and early access testing**. A full installation guide and tutorial will be available soon.

---

## 🚀 Installation
CellLENS is hosted on `pypi` and can be installed via `pip`. We recommend working within a virtual environment. The package requires `CUDA` support as it uses `PyTorch`.

```bash
conda create -n celllens python=3.9   # Create a new environment
conda activate celllens               # Activate environment
pip install celllens           # Install CellLENS
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

📌 **[Tutorial II.v2 - quick Xenium example](https://github.com/sggao/celllens/blob/main/tutorials/Tutorial_Full_CellLENS(Xenium).ipynb)** - Similar to Tutorial II but using Xenium data as an example.

📌 **[Tutorial III](https://github.com/sggao/celllens/blob/main/tutorials/Optionl_Tutorial_Full_CellLENS-ViT.ipynb)** - Advanced usage with **ViT-based** image feature learning.

---

## ⚙️ Key Parameters & Recommendations

| Name               | Function             | Description  | Recommendation |
|-------------------|----------------------|------------------|----------------|
| `nbhd_composition` | `LENS_Dataset()` | Number of nearest neighbors (cells) to consider when calculating the ‘neighborhood composition vector’. This vector is involved in the CellLENS training to learn local cellular patterns. <br> **Default = 20** | Generally no need to change this parameter. Could tune from 10-50, depending on the scale of the local cellular pattern wanted to learn in the tissue. |
| `feature_neighbor` | `LENS_Dataset()` | Number of nearest neighbors to consider when linking the nodes (cells) on the feature (eg. protein expression) similarity graph for the GNN training. <br> **Default = 15** | No need to change this parameter in most cases. |
| `spatial_neighbor` | `LENS_Dataset()` | Number of nearest neighbors to consider when linking the nodes (cells) on the spatial (location) similarity graph for the GNN training. <br> **Default = 15** | No need to change this parameter in most cases. |
| `pca_components` | `.initialize()` | Number of components to use on the PCA reduced feature expression input. <br> **Default = 25** | Users should decide the value here. This is similar to the conventional PCA selection process, for example scRNA-seq studies. |
| `celltype` | `.initialize()` | Column to use as initial cell type labels. If input is `'feature_labels'` then will use Leiden clustering to get the initial labels; Alternatively, the user can also supply the pre-generated cell type labels from other means (eg. previous annotation, label transfer, and more). <br> **Default = ‘feature_labels’** | If no pre-generated cell type information is available, the user can use the default; If the coarse cell type information is available, the user can supply here. |
| `cluster_res` | `.initialize()` | The resolution parameter for Leiden clustering will determine the initial labels used to calculate the 'neighborhood composition vector,' which serves as input for the CellLENS model learning process. Alternatively, users may opt to specify a fixed number of clusters instead of setting a resolution (see details below). <br> **Default = 0.5** | Resolution of 0.5 works in most cases. However, this might be influenced when the dataset is too large or too small. We suggest the user monitor the verbose print-outs during this step. Generally, initial labels of 8-15 types should yield optimized results. |
| `n_clusters` | `.initialize()` | Number of clusters to generate during Leiden clustering. If None will use the specified resolution to run Leiden clustering instead. <br> **Default = None** | Same description as above. The user can choose to specify the number of initial label types to be generated during Leiden clustering. We suggest around 8-15 clusters. |
| `size` | `.prepare_images()` | The pixel size to be considered when cropping the image for each individual cell. The image will then be used to extract local tissue level morphological information in the CNN model. <br> **Default = 512** | Users should decide the value here, since the pixel’s physical distance could differ from modality to modality. Generally, a translated physical distance of 50 - 200 μm for the image size works well. |
| `truncation` | `.prepare_images()` | Pixel intensity quantile as threshold to binarize (0,1) the images input in the CNN model. Pixels with intensity level larger than this quantile will be set as 1, and pixels smaller than this quantile will be set as 0. <br> **Default = 0.9** | Generally, we found 0.7-0.9 quantile works well. The users can visually check the images with different quantiles before running the CNN model. |
| `cnn_latent_dim` | `CellLENS()` | The size of the latent dimension (extracted image features) from the CNN model. <br> **Default = 128** | No need to change this parameter in most cases. |
| `gnn_latent_dim` | `CellLENS()` | The size of the latent dimension (extracted fused representation) from the duo-GNN model. <br> **Default = 32** | No need to change this parameter in most cases. |
| `fc_out_dim` | `CellLENS()` | Output dimension for expression-GNN and input to MLP head. Larger value of this parameter encourages the model to learn more expression-related information. <br> **Default = 33** | No need to change this parameter in most cases. |
| `cnn_out_dim` | `CellLENS()` | Output dimension for spatial-GNN and input to MLP head. Larger value of this parameter encourages the model to learn more image morphology-related information. <br> **Default = 11** | No need to change this parameter in most cases. |
| `round` | `.get_lens_embedding()` | Number of times the CellLENS duo-GNN model is trained. The repeated training is to produce a robust final representation. <br> **Default = 5** | No need to change this parameter in most cases. |
| `k` | `.get_lens_embedding()` | We run SVD on the embedding generated by repeated rounds of duo-GNN training, and retrieve the k dimensions of the SVD results as the final representation. <br> **Default = 32** | No need to change this parameter in most cases. |


## 📢 Citation
If you find **CellLENS** useful, please cite our [paper](https://www.nature.com/articles/s41590-025-02163-1).
```
@article{Zhu2025CellLENS,
  author = {Zhu, Bokai and Gao, Sheng and Chen, Shuxiao and Wang, Yuchen and Yeung, Jason and Bai, Yunhao and Huang, Amy Y. and Yeo, Yao Yu and Liao, Guanrui and Mao, Shulin and Jiang, Zhenghui G. and Rodig, Scott J. and Wong, Ka-Chun and Shalek, Alex K. and Nolan, Garry P. and Jiang, Sizun and Ma, Zongming},
  title = {CellLENS enables cross-domain information fusion for enhanced cell population delineation in single-cell spatial omics data},
  journal = {Nature Immunology},
  volume = {26},
  pages = {963--974},
  year = {2025},
  doi = {10.1038/s41590-025-02163-1}
}
```

---

## 📬 Contact & Contributions
We welcome contributions and feedback! Please open an [issue](https://github.com/sggao/celllens/issues) or submit a pull request.

---
