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
