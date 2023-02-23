from openslide import OpenSlide
import math
from anndata import AnnData
import squidpy as sq
import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
from numpy.random import default_rng
import seaborn as sns
import warnings 
from utils import get_class, get_color_ids
import pdb

def gen_output(results):
    coord = [c for sublist in results['coordinates'] for c in sublist]
    x = [c[0].detach().cpu().numpy() for c in coord]
    y = [c[1].detach().cpu().numpy() for c in coord]
    labels = [l for sublist in results['label'] for l in sublist]
    df_res = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    return df_res

def get_matracies(slide, cluster_df, patch_size = (112, 112)):

    if not slide.properties.get('openslide.mpp-x'): print(f"resolution is not found, using default 0.5um/px")
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster_df['new_x'] = np.ceil(cluster_df['x'].values / patch_size_resized[0])
        cluster_df['new_y'] = np.ceil(cluster_df['y'].values / patch_size_resized[1])

        cluster_df['new_x'] = cluster_df['new_x'].astype(int)
        cluster_df['new_y'] = cluster_df['new_y'].astype(int)

    matrix_trait = pd.DataFrame({'label': cluster_df['label'], 'x':  cluster_df['new_x'], 'y': cluster_df['new_y']})

    return(matrix_trait)


def gen_graph(slide, results):

    cluster_df = gen_output(results)

    trait = get_matracies(slide, cluster_df = cluster_df, patch_size = (112, 112))
    trait['label'] = trait['label'].astype('category')
    class2idx, id2class = get_class()
    labels = sorted(np.unique(trait['label']))
    cell_types = [id2class[k] for k in labels]

    cell_number = trait.shape[0]
    rng = default_rng(0)                            
    counts = rng.integers(0, 15, size=(cell_number, 50))  # feature matrix
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = AnnData(counts, obs = trait, obsm={"spatial": np.asarray(trait[['x', 'y']])}, dtype = counts.dtype)

    sq.gr.spatial_neighbors(adata, n_neighs=8, n_rings=2, coord_type="grid")
    sq.gr.centrality_scores(adata, cluster_key='label', show_progress_bar=False)
    sq.gr.interaction_matrix(adata, cluster_key='label', normalized = True)

    # Generate dataframes
    dgr_centr = pd.DataFrame({'cell_type': cell_types, 'centrality':adata.uns['label_centrality_scores']['degree_centrality']})
    im_mtx = pd.DataFrame(adata.uns['label_interactions'], columns=cell_types, index=cell_types)
    cluster_res = []
    for cell in cell_types:
        cluster_res.append(im_mtx.loc[cell][cell])
    df_cluster = pd.DataFrame({'cell_type': cell_types, 'cluster_coeff': cluster_res})

    dgr_centr = dgr_centr.sort_values(["centrality"], ascending=False)
    df_cluster = df_cluster.sort_values(['cluster_coeff'], ascending=False)

    dgr_centr = dgr_centr.reset_index(drop = True)
    df_cluster = df_cluster.reset_index(drop = True)

    return dgr_centr, im_mtx, df_cluster

def compute_percent(labels):
    labels = labels['label']
    labels = np.concatenate((labels), axis=0)
    # convert predicted labels to actual cell types
    class2idx, id2class = get_class()
    pred_labels = [id2class[k] for k in labels]
    total = len(pred_labels)
    cell_types = class2idx.keys()
    frac = []
    for cell in cell_types:
        count = pred_labels.count(cell)
        percent = float(count/total) * 100
        frac.append(percent)
    df = pd.DataFrame({'cell_type': cell_types, 'Percentage': frac})
    df = df.sort_values(['Percentage'], ascending=False)
    df = df.reset_index(drop=True)
    return df


    
        



    





    






    plt.figure()
    sq.pl.interaction_matrix(adata, cluster_key=args.label)
    plt.savefig(os.path.join(inter_matrix_path, f"{s}.png"), bbox_inches='tight')
    plt.close()



    


