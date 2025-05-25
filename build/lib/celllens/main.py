import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import skimage
import sys

sys.path.append("../src/celllens/")
from .utils import *
import os
from tqdm import tqdm
from skimage.io import imread
from .preprocessing import *
from .datasets import *
from .celllens import *

#############################
# example code in this file #
#############################


def main():
    # pipeline for codex murine dataset
    df = pd.read_csv('data/codex_murine/features_and_metadata.csv',
                     index_col=0)
    # might want to preprocess
    df.fillna(0, inplace=True)
    features_list = [
        'CD45', 'Ly6C', 'TCR', 'Ly6G', 'CD19', 'CD169', 'CD106', 'CD3',
        'CD1632', 'CD8a', 'CD90', 'F480', 'CD11c', 'Ter119', 'CD11b', 'IgD',
        'CD27', 'CD5', 'CD79b', 'CD71', 'CD31', 'CD4', 'IgM', 'B220', 'ERTR7',
        'MHCII', 'CD35', 'CD2135', 'CD44', 'nucl', 'NKp46'
    ]
    murine_dataset = LENS_Dataset(
        df,
        features_list=features_list,
        nbhd_composition=15,
        feature_neighbor=15,
        spatial_neighbor=15,
        path2img='../../data/tutorial/codex_murine/processed_images')
    # prepare meta data
    murine_dataset.initialize(cent_x="centroid_x",
                              cent_y="centroid_y",
                              celltype="feature_labels",
                              pca_components=25,
                              cluster_res=1.0)

    # train CellLENS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    murine_celllens = CellLENS(murine_dataset,
                               device,
                               cnn_model=True,
                               cnn_latent_dim=128,
                               gnn_latent_dim=32)
    # Load pretrained LENS-CNN embedding
    murine_celllens.cnn_embedding = np.load(
        'data/codex_murine/results/LENS_CNN_embedding.npy')
    murine_celllens.get_lens_embedding(round=3,
                                       k=32,
                                       learning_rate=1e-3,
                                       n_epochs=5000,
                                       loss_fn='MSELoss',
                                       OptimizerAlg='Adam',
                                       optimizer_kwargs={},
                                       SchedulerAlg=None,
                                       scheduler_kwargs={},
                                       verbose=True)
    # clustering and visualization
    murine_celllens.get_lens_clustering(neighbor=15, resolution=1.0)
    murine_celllens.visualize_umap(murine_celllens.lens_embedding,
                                   murine_celllens.lens_clustering)


if __name__ == '__main__':
    main()
