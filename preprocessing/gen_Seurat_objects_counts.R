# Convert raw 10X output to Seurat objects
library(Seurat)

sample_list <- c('242_T', '243_T', '248_T', '251_T', '255_T', '256_T', '259_T',
                 '260_T', '262_T', '265_T', '266_T', '268_T', '269_T', '270_T',
                 '275_T', '296_T', '304_T', '313_T', '334_T')

data_dir <- "spatial_brain/data/raw" # directory for raw 10X output
save_dir <- "spatial_brain/data/Seurat_object_h5" # save path

for(sample in sample_list){
  print(sample)
  data.dir <- fileh5 <-  NULL
  data.dir <- paste0(data_dir, "/", "#UKF", sample, "/", sample)
  fileh5 <- paste0(data.dir, "/filtered_feature_bc_matrix.h5")
  Seurat_obj <- CreateSeuratObject(
    counts = Read10X_h5(filename = fileh5),
    assay = 'Spatial',
    min.cells = 3,
    min.features = 200
  )
  saveRDS(Seurat_obj, paste0(save_dir, "/", sample, ".rds"))
}
