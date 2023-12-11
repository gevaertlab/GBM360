import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import infercnvpy
from genepos import genomic_position_from_gtf
import pdb

source_dir = "."
data_dir = os.path.join(source_dir, "data/Spatial_Heiland/data/AnnDataObject")
gtf_file = os.path.join(source_dir,"data/Ref_Genome/gencode.v43.annotation.txt")
res_dir = os.path.join(source_dir, 'data/Spatial_Heiland/results/cnv')

print("=======Reading in data============")
adata = sc.read_h5ad(os.path.join(data_dir, 'concat_counts_three_data.h5ad'))
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

print("=======Inferring CNV============")
gtf = pd.read_csv(gtf_file, sep = "\t")
genomic_position_from_gtf(gtf, adata=adata, gtf_gene_id='gene_name')
infercnvpy.tl.infercnv(adata, reference_key = 'dataset', reference_cat = 'normal')
sc.write(os.path.join(data_dir, 'cnv_with_normal.h5ad'), adata)