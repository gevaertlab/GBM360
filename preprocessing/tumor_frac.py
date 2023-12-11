# Visulize the heatmap of CNAs and infer tumor fraction 
import scanpy as sc
import anndata as ad
from anndata import AnnData
import pandas as pd
import numpy as np
import scipy.sparse
import os
import matplotlib.pyplot as plt
import infercnvpy
import seaborn as sns
import pdb

source_dir = "."
data_dir = os.path.join(source_dir, "data/Spatial_Heiland/data/AnnDataObject")
save_dir = os.path.join(source_dir, "data/Spatial_Heiland/results/cnv")
plot_dir = os.path.join(save_dir, "heatmap")

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

adata = sc.read_h5ad(os.path.join(data_dir, "cnv_with_normal.h5ad"))

# Filter out some low-quality samples
adata = adata[~adata.obs['slide_id'].isin(["256_TC", "265_T", "256_TI"])]
adata_normal = adata[adata.obs['dataset'] == "normal"]
adata_tumor = adata[adata.obs['dataset'] != "normal"]

# All data
plt.figure()
infercnvpy.pl.chromosome_heatmap(adata, groupby="slide_id", figsize = (12,12))
plt.savefig(os.path.join(plot_dir, "cnv_heatmap_all.png"))

# Tumor data
plt.figure()
infercnvpy.pl.chromosome_heatmap(adata_tumor, groupby="slide_id", figsize = (12,12))
plt.savefig(os.path.join(plot_dir, "cnv_heatmap_tumor.png"))

# Normal data
plt.figure()
infercnvpy.pl.chromosome_heatmap(adata_normal, groupby="slide_id", figsize = (12,12))
plt.savefig(os.path.join(plot_dir, "cnv_heatmap_normal.png"))

use_rep = "cnv"
df_cnv = pd.DataFrame.sparse.from_spmatrix(adata.obsm[f'X_{use_rep}'])
df_cnv.index = adata.obs.index

# Retrieve chromosomal region 
chr_pos_dict = dict(sorted(adata.uns[use_rep]["chr_pos"].items(), key=lambda x: x[1]))
chr_pos = list(chr_pos_dict.values())
var_group_positions = list(zip(chr_pos, chr_pos[1:] + [adata.shape[1]]))

# Add the chr id as prefix to each chromosomal region
new_cols =[]
for col in list(df_cnv.columns):
    for i, var in enumerate(var_group_positions):
        if col >= var[0] and col < var[1]:
            new_col = f"chr{i+1}:{col}"
            new_cols.append(new_col)
            continue
df_cnv.columns = new_cols

# Split normal and tumor matrix
adata.obsm['cnv_chr'] = df_cnv
adata_normal = adata[adata.obs['dataset'] == "normal"]
adata_tumor = adata[adata.obs['dataset'] != "normal"]

# Check the cnv scores of normal tissues for each region
avg_normal = adata_normal.obsm['cnv_chr'].mean(axis = 0)
print(np.min(avg_normal), np.max(avg_normal), np.mean(avg_normal))

all_res = []

# Calclulate tumor cell fraction for each tissue
slides = list(adata_tumor.obs['slide_id'].unique())
slides = ['334_T']

for slide in slides:
    print(slide)
    cur_ann = adata[adata.obs['slide_id'] == slide]
    cur_cnv = cur_ann.obsm['cnv_chr']

    # sort the chr regions based on their absolute mean values of cnvs
    abs_mean = cur_cnv.apply(lambda x: x.abs().mean())
    sorted_mean = abs_mean.sort_values(ascending=False)
    cur_cnv = cur_cnv.loc[:, sorted_mean.index]

    # select only the chr regions with absolute mean values greater than 0.02
    cur_cnv = cur_cnv.loc[:, abs(sorted_mean) > 0.025]

    # select only the top 15 signiatures
    if cur_cnv.shape[1] > 15:
        cur_cnv = cur_cnv.iloc[:, 0:15]

    # group chr regions by chr
    cnv_grouped = cur_cnv.groupby(lambda x: x.split(':')[0], axis=1).agg('mean')
    cnv_grouped = cnv_grouped + 1

    # calculate cell frac based on cnvs of each chr
    def cal_frac(x):
        if x.mean() > 1:
            frac = (x-1)/(x.max()-1)
        else:
            frac = (1-x)/(1-x.min())
        return frac

    cnv_frac = cnv_grouped.apply(lambda x: cal_frac(x))
    cell_frac = cnv_frac.max(axis = 1)
    normal_spots = cell_frac[cell_frac<0.2].index
    tumor_spots = cell_frac[cell_frac>=0.2].index
    df_frac = pd.DataFrame(cell_frac, columns=['tumor_cell_frac'])
    df_frac['spot_type'] = np.where(df_frac['tumor_cell_frac'] < 0.2, "normal", "malignant")
    all_res.append(df_frac)

    # plot heatmap of tumor cell fractions
    cur_ann.obs = cur_ann.obs.merge(df_frac, left_index = True, right_index= True)

    plt.figure()
    sc.pl.spatial(cur_ann, 
                color=["tumor_cell_frac", "spot_type"], 
                library_id = slide,
                #vmin = [None, 0.1], vmax = [None, 0.8],
                cmap="RdYlBu_r", 
                palette = {'normal': sns.color_palette()[1], 'malignant': sns.color_palette()[0]}, 
                legend_fontsize = 'large', 
                alpha_img = 0.3, 
                size = 1.5, 
                title = '',
                legend_loc='right margin',
                na_in_legend = False)
    plt.savefig(os.path.join(plot_dir, f'{slide}.eps'))

df_final = pd.concat(all_res)
df_final.to_csv(os.path.join(save_dir,  "cnv_status.csv"))