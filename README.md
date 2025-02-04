# CellLENS

[![Python Versions](https://img.shields.io/pypi/pyversions/cellsnap.svg)](https://pypi.org/project/cellsnap)
[![PyPI Version](https://img.shields.io/pypi/v/cellsnap.svg)](https://pypi.org/project/cellsnap)
[![GitHub Issues](https://img.shields.io/github/issues/sggao/cellsnap.svg)](https://github.com/sggao/cellsnap/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sggao/cellsnap/blob/master/tutorials/CellSNAP_codex_murine.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<img src="https://github.com/sggao/CellLENS/blob/main/media/figure1_v4.png">

## Description
Official implementation of Cell Local Environment Neighborhood Scan (CellLENS), a computational method that learns a single-cell representation embedding by integrating cross-domain information from tissue samples.
Through the analysis of datasets spanning spatial proteomic and spatial transcriptomic modalities, and across different tissue types and disease settings, we demonstrate CellLENS’s capability to elucidate biologically relevant cell populations that were previously elusive due to the relinquished tissue morphological information from images. For more details, see our [preprint](https://www.biorxiv.org/content/10.1101/2024.05.12.593710v1).


NOTE: this repository is under active development, and the current version is only meant for <ins>reviewing and early access testing etc</ins>. We will provide more detailed installation instruction and tutorial soon.

## Installation
CellLENS is hosted on `pypi` and can be installed via `pip`. We recommend working with a fresh virtual environment. In the following example we use conda. Note our package requires the availability of `CUDA` on the machine, since it uses `pyTorch`.

```
conda create -n celllens python=3.9 # create a new vm
conda activate celllens # activate celllens vm
pip install celllens==0.1.0 # install celllens in vm
```

After installation, you can import the module via
```
import celllens
```
For details of the use of CellLENS, please refer to the tutorials.

## Tutorials

For simplified CellLENS usage, please see [Tutorial I](https://github.com/sggao/celllens/blob/main/tutorials/Tutorial_LITE_CellLENS.ipynb). This tutorial uses both feature expression and cell locations.

For full version CellLENS usage, please see [Tutorial II]([https://github.com/sggao/celllens/blob/main/tutorials/Tutorial_LITE_CellLENS.ipynb](https://github.com/sggao/celllens/blob/main/tutorials/Tutorial_Full_CellLENS.ipynb)). This tutorial uses feature expression, cell locations, and tissue images.

For development reasons, if you are interested in more complex architecture during CellLENS image feature learning processes, please see [Tutorial III](https://github.com/sggao/celllens/blob/main/tutorials/Optionl_Tutorial_Full_CellLENS-ViT.ipynb). This CellLENS version uses a ViT architecture during image feature learning.

## 📌 Key Parameters and Recommendations

| Parameter          | Function                | Description  | Default | Recommendation |
|-------------------|----------------------|------------------|---------|----------------|
| `nbhd_composition` | `SNAP_Dataset()` | Number of nearest neighbors (cells) to consider when calculating the ‘neighborhood composition vector’. This vector is involved in the CellLENS training to learn local cellular patterns. | 20 | Generally no need to change this parameter. Could tune from 10-50, depending on the scale of the local cellular pattern. |
| `feature_neighbor` | `SNAP_Dataset()` | Number of nearest neighbors to consider when linking nodes (cells) on the feature (e.g., protein expression) similarity graph for GNN training. | 15 | No need to change in most cases. |
| `spatial_neighbor` | `SNAP_Dataset()` | Number of nearest neighbors to consider when linking nodes (cells) on the spatial (location) similarity graph for GNN training. | 15 | No need to change in most cases. |
| `pca_components` | `.initialize()` | Number of components to use on the PCA reduced feature expression input. | 25 | Similar to conventional PCA selection, e.g., in scRNA-seq studies. User should decide. |
| `celltype` | `.initialize()` | Column to use as initial cell type labels. If ‘feature_labels’, Leiden clustering will generate the labels, otherwise, the user can provide pre-defined labels. | `feature_labels` | If no pre-generated cell type information is available, use the default; otherwise, supply known labels. |
| `cluster_res` | `.initialize()` | Resolution parameter for Leiden clustering to determine initial labels for CellLENS model input. | 0.5 | Works in most cases. Adjust if dataset is too large or small; monitor verbose printouts. Generally, 8-15 types yield optimal results. |
| `n_clusters` | `.initialize()` | Number of clusters to generate during Leiden clustering. If `None`, resolution is used instead. | None | User can specify 8-15 clusters instead of resolution. |
| `size` | `.prepare_images()` | Pixel size for cropping individual cell images used in CNN-based local tissue morphology analysis. | 512 | Depends on physical scale in modality. Typically, 50-200 μm works well. |
| `truncation` | `.prepare_images()` | Pixel intensity quantile for binarizing (0,1) CNN model inputs. | 0.9 | Typically, 0.7-0.9 works well. Users should visually check images before proceeding. |
| `cnn_latent_dim` | `CellSNAP()` | Size of the latent dimension (extracted image features) from CNN. | 128 | No need to change in most cases. |
| `gnn_latent_dim` | `CellSNAP()` | Size of the latent dimension (extracted fused representation) from the duo-GNN model. | 32 | No need to change in most cases. |
| `fc_out_dim` | `CellSNAP()` | Output dimension for expression-GNN, used as MLP head input. Larger values emphasize expression-related features. | 33 | No need to change in most cases. |
| `cnn_out_dim` | `CellSNAP()` | Output dimension for spatial-GNN, used as MLP head input. Larger values emphasize morphological features. | 11 | No need to change in most cases. |
| `round` | `.get_snap_embedding()` | Number of training rounds for the CellLENS duo-GNN model. Repeated training ensures robust representation. | 5 | No need to change in most cases. |
| `k` | `.get_snap_embedding()` | SVD is run on embeddings from repeated duo-GNN training rounds. This defines the final representation's dimensions. | 32 | No need to change in most cases. |

---

