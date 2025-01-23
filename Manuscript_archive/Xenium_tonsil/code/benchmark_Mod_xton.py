#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import anndata as ad 
import scanpy as sc
sys.path.append("../../../../")
import utils
from sklearn.decomposition import PCA
import umap
import graph
import visualize
import sklearn.metrics


# In[2]:


# define modularity score

import leidenalg
import igraph as ig

def get_modularity(n,edges, seed = 42, resolution = 1):
    
    g = ig.Graph(directed=True)
    g.add_vertices(n)
    g.add_edges(list(zip(edges[0], edges[1])))
    g.es['weight'] = tuple(edges[2])
    partition_kwargs = {'n_iterations': -1, 'seed': seed,
                    'resolution_parameter': resolution}
    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    partition_type = leidenalg.RBConfigurationVertexPartition
    curr_part = leidenalg.find_partition(
                graph=g, partition_type=partition_type,
                **partition_kwargs
            )
    curr_modularity = curr_part.modularity
    
    return curr_modularity


# In[3]:


# read in files
snap_embed = np.load('../data/Xton_double_snap_combo_stable.npy')
features = pd.read_csv('../data/Xton_sub_pc.csv').iloc[:,:30].to_numpy()
concact_embed = np.load("../data/xton_concact.npy")
spice_embed = np.load("../data/spicemix_xton_embedding.npy")

##### load in muse 0-4 batch results
muse_embed0 = np.load("../data/muse_xton_embedding_V2batch_0.npy")
muse_embed1 = np.load("../data/muse_xton_embedding_V2batch_1.npy")
muse_embed2 = np.load("../data/muse_xton_embedding_V2batch_2.npy")
muse_embed3 = np.load("../data/muse_xton_embedding_V2batch_3.npy")
muse_embed4 = np.load("../data/muse_xton_embedding_V2batch_4.npy")
muse_embedll = [muse_embed0, muse_embed1, muse_embed2, muse_embed3, muse_embed4]

###
cca = pd.read_csv('../data/cca8_feat.csv').to_numpy()
mofa = pd.read_csv('../data/mofa_fc25.csv').to_numpy()


print([snap_embed.shape, features.shape, concact_embed.shape,
      spice_embed.shape, muse_embed0.shape, cca.shape, mofa.shape])


# In[4]:


# for loop start:
batch = 5
dflist = []
for i in range(batch):
    print('at', i)
    
    indices = np.random.choice(snap_embed.shape[0], 10000, replace=False)
    
    snap_embed_sub = snap_embed[indices]
    features_sub = features[indices]
    concact_embed_sub = concact_embed[indices]
    spice_embed_sub = spice_embed[indices]
    muse_embed_sub = muse_embedll[i]
    cca_sub = cca[indices]
    mofa_sub = mofa[indices]
    
    ##### calculate umap
    
    # start to calculate silhoutte score
    feature_ss_list = []
    snap_ss_list = []
    concact_ss_list = []
    spice_ss_list = []
    muse_ss_list = []
    cca_ss_list = []
    mofa_ss_list = []
    
    ## feature edges
    feature_edges = graph.get_feature_edges(
        arr=features_sub, pca_components=None,
        n_neighbors=15, metric='correlation', verbose=False
    )

    ## snap
    snap_edges = graph.get_feature_edges(
        arr=snap_embed_sub, pca_components=None,
        n_neighbors=15, metric='correlation', verbose=False
    )

    ## concact 
    concact_edges = graph.get_feature_edges(
        arr=concact_embed_sub, pca_components=None,
        n_neighbors=15, metric='correlation', verbose=False
    )

    ## muse 
    muse_edges = graph.get_feature_edges(
        arr=muse_embed_sub, pca_components=None,
        n_neighbors=15, metric='correlation', verbose=False
    )

    ## spice 
    spice_edges = graph.get_feature_edges(
        arr=spice_embed_sub, pca_components=None,
        n_neighbors=15, metric='correlation', verbose=False
    )

    ## cca 
    cca_edges = graph.get_feature_edges(
            arr=cca_sub, pca_components=None,
            n_neighbors=15, metric='correlation', verbose=False
    )
    
    ## mofa
    mofa_edges = graph.get_feature_edges(
            arr=mofa_sub, pca_components=None,
            n_neighbors=15, metric='correlation', verbose=False
    )
    

    res_list = [round(x, 2) for x in np.arange (0.4, 2.6, 0.2)]
    for res in res_list:
        
        feature_ss = get_modularity(features_sub.shape[0], feature_edges, resolution = res)
        snap_ss = get_modularity(snap_embed_sub.shape[0], snap_edges, resolution = res)
        concact_ss = get_modularity(concact_embed_sub.shape[0], concact_edges, resolution = res)
        spice_ss = get_modularity(spice_embed_sub.shape[0], spice_edges, resolution = res)
        muse_ss = get_modularity(muse_embed_sub.shape[0], muse_edges, resolution = res)
        cca_ss = get_modularity(cca_sub.shape[0], cca_edges, resolution = res)
        mofa_ss = get_modularity(mofa_sub.shape[0], mofa_edges, resolution = res)
        
        feature_ss_list.append(feature_ss)
        snap_ss_list.append(snap_ss)
        concact_ss_list.append(concact_ss)
        spice_ss_list.append(spice_ss)
        muse_ss_list.append(muse_ss)
        cca_ss_list.append(cca_ss)
        mofa_ss_list.append(mofa_ss)


    data = {'res':res_list, 'ch_feature':feature_ss_list,
            'ch_snap':snap_ss_list,'ch_concact':concact_ss_list,
           'ch_spice':spice_ss_list, 'ch_muse':muse_ss_list,
           'ch_cca':cca_ss_list, 'ch_mofa':mofa_ss_list}
    
    df = pd.DataFrame(data)
    df['batch'] = i
    dflist.append(df)
    


# In[5]:


test = pd.concat(dflist)
test2 = test.melt(id_vars=['res'], value_vars=['ch_feature', 'ch_snap', 'ch_concact', 'ch_spice', 'ch_muse',
                                             'ch_cca', 'ch_mofa'])
test2.to_csv('../data/Mod_result_xton.csv')

plot = sns.lineplot(data=test2,x="k", y="value", hue="variable")
fig = plot.get_figure()
fig.savefig('../plots/Mod_allmethods.svg', dpi = 300) 


# In[ ]:





# In[ ]:




