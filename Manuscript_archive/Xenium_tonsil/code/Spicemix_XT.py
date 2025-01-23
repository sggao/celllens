import time, os, sys, pickle, h5py, importlib, gc, copy, re, itertools, json, logging
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
from tqdm.auto import tqdm, trange
from pathlib import Path
from util import config_logger
from util import openH5File, encode4h5, parse_suffix, config_logger


import numpy as np, pandas as pd, scipy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from umap import UMAP
import anndata as ad

import scanpy as sc

import torch
torch.set_num_threads(16)
from model import SpiceMix
from helper import evaluate_embedding_maynard2021 # This function is for the optional on-the-fly evaluation. This is not required for SpiceMix.
fn_eval = evaluate_embedding_maynard2021

from matplotlib import pyplot as plt
from load_data import load_expression, load_edges, load_genelist

import seaborn as sns
sns.set_style("white")
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

logger = config_logger(logging.getLogger(__name__))

df_counts = pd.read_csv('../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/Xton_sub_norm.csv')
gn = pd.read_csv('../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/gene_names.csv')
df_counts.columns = gn.iloc[:,0].to_list()
df_counts.head()

cellpop = np.load('../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/Xton_feature_labels_res0.5.npy')
df_counts['cellpop'] = cellpop

save_path = "../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/spicemix/files/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

df_meta = pd.read_csv('../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/Xton_sub_meta_snap_feat_clusters.csv')
df_meta.head()


features_names = df_counts.columns

#np.savetxt(os.path.join(save_path, f"expression_chl.txt"), df_counts.loc[:50000,features_names]) #####
np.savetxt(os.path.join(save_path, f"expression_xton.txt"), df_counts.loc[:,features_names])

# save neighborhood
locations = df_meta.loc[:,['x_centroid', 'y_centroid']].to_numpy()
adata = ad.AnnData(df_counts, dtype=np.float32)
adata.obsm["spatial"] = locations
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=None, use_rep='spatial')
rows, cols = adata.obsp['connectivities'].nonzero()
edges = np.array([rows, cols]).T

#file = open(os.path.join(save_path, f"neighborhood_chl.txt"),'w') ######
file = open(os.path.join(save_path, f"neighborhood_xton.txt"),'w')
for i in range(rows.shape[0]):
    file.write(str(rows[i])+" "+str(cols[i])+"\n")
file.close()

# save cell type
file = open(os.path.join(save_path, f"genes_xton.txt"),'w') #####

for item in features_names:
    file.write(item+"\n")
file.close()

file = open(os.path.join(save_path, f"celltypes_xton.txt"),'w') # note this does not have human intervention annotaion
for i in range(df_counts.shape[0]):
    file.write(df_counts.loc[:,'cellpop'].astype(str).to_list()[i] + "\n")
file.close()

# -- specify device
context = dict(device='cpu', dtype=torch.float64)
# context = dict(device='cpu', dtype=torch.float64)
context_Y = context
# -- specify dataset
path2dataset = Path('../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/spicemix/')
#repli_list = list(df['fov'].unique().astype(str))
repli_list = ['xton']


from model import SpiceMix
from helper import evaluate_embedding_maynard2021 # This function is for the optional on-the-fly evaluation. This is not required for SpiceMix.
fn_eval = evaluate_embedding_maynard2021

np.random.seed(0)

K, num_pcs, n_neighbors, res_lo, res_hi = 20, 20, 15, .5, 2.

path2result = path2dataset / 'results' / 'SpiceMix.h5'
os.makedirs(path2result.parent, exist_ok=True)
if os.path.exists(path2result):
    os.remove(path2result)
    
obj = SpiceMix(
    K=K,
    lambda_Sigma_x_inv=1e-6, power_Sigma_x_inv=2,
    repli_list=repli_list,
    context=context,
    context_Y=context,
    path2result=path2result,
)
obj.load_dataset(path2dataset)


obj.meta['cell type'] = pd.Categorical(obj.meta['cell type'])
obj.initialize(
    method='louvain', kwargs=dict(num_pcs=num_pcs, n_neighbors=n_neighbors, resolution_boundaries=(res_lo, res_hi), num_rs = 2),
)
for iiter in range(200):
    obj.estimate_weights(iiter=iiter, use_spatial=[False]*obj.num_repli)
    obj.estimate_parameters(iiter=iiter, use_spatial=[False]*obj.num_repli)
obj.initialize_Sigma_x_inv()


latent_states = [X.cpu().numpy() for X in obj.Xs]
latent_state_cat = np.concatenate(latent_states, axis=0)


np.save('../../spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/tonsil_xenium/data/spicemix_xton_embedding.npy', latent_state_cat) 
print('done')


