library(Seurat)
library(dplyr)

data.dir <- "spatial_brain/data/Seurat_object_h5"
save.dir <- "spatial_brain/result/cluster"
cnv <- read.csv("spatial_brain/cnv/cnv_status_final.csv")
colnames(cnv) <- c("barcode", "cnv_status")

sample_list <- c('242_T', '243_T', '248_T', '251_T', '255_T', '256_T', '259_T',
                 '260_T', '262_T', '265_T', '266_T', '268_T', '269_T', '270_T',
                 '275_T', '296_T', '304_T', '313_T', '334_T')

object_list <- c()
num_spots_total <- c()
num_spots_tumor <- c()
num_spots_after <- c()
processed_samples <- c()

for(sample in sample_list){
  if(sample %in% processed_samples) next
  data <- markers <- merged.data <- NULL
  print(paste0("Reading ", sample))
  data <- readRDS(paste0(data.dir, "/", sample, ".rds"))
  data[["orig.ident"]] <- sample
  data[["barcode"]] <- paste0(rownames(data[[]]), "_", sample)
  print(paste0("Number of spots: ", nrow(data[[]])))
  num_spots_total <- c(num_spots_total, nrow(data[[]]))

  # Add tumor or normal information
  merged.data <- merge(x=data[[]], y= cnv, by="barcode", all.x=TRUE)
  rownames(merged.data) = rownames(data[[]])
  stopifnot(identical(merged.data$barcode, data[[]]$barcode)) # safety check
  data@meta.data <- merged.data
  print(paste0("Number of tumor spots: ", table(data@meta.data$cnv_status)['tumor']))
  num_spots_tumor <- c(num_spots_tumor, table(data@meta.data$cnv_status)['tumor'])

  print("Fitering QC")
  data <- PercentageFeatureSet(data, pattern = "^MT-", col.name = "percent.mt")
  data <- subset(data, subset = nCount_Spatial >= 1000 & nFeature_Spatial >= 200 & percent.mt <= 5 & cnv_status != "normal")
  if(!sample %in% c("256_T", "262_T")){
    data <- CellCycleScoring(data, s.features = cc.genes$s.genes,
                             g2m.features = cc.genes$g2m.genes,
                             set.ident = TRUE)
    data <- SCTransform(data,  assay = "Spatial",
                        variable.features.n = 3000,
                        vars.to.regress = c("percent.mt", "S.Score", "G2M.Score"),
                        verbose = FALSE)
  }else{
    data <- SCTransform(data,  assay = "Spatial",
                        variable.features.n = 3000,
                        vars.to.regress = c("percent.mt"),
                        verbose = FALSE)
  }

  print(paste0("Number of spots after QC: ", nrow(data[[]])))
  num_spots_after <- c(num_spots_after, nrow(data[[]]))

  object_list <- c(object_list, data)
  processed_samples <- c(processed_samples, sample)
}

sample_sum <- data.frame(sample_id = processed_samples, num_spots_total = num_spots_total,
                         num_spots_tumor = num_spots_tumor, num_spots_after = num_spots_after)

write.csv(sample_sum, paste0(save.dir, "/", "sample_spot_summary.csv"))

object_list <- object_list[-6] #remove 256_T where most of the cells are normal

# Data integration
features <- SelectIntegrationFeatures(object.list = object_list, nfeatures = 3000)
object_list <- PrepSCTIntegration(object.list = object_list, anchor.features = features)
object_list <- lapply(X = object_list, FUN = RunPCA, features = features)
anchors <- FindIntegrationAnchors(object.list = object_list, normalization.method = "SCT",
                                  anchor.features = features, dims = 1:30, reduction = "rpca", k.anchor = 20)
integrated.data <- IntegrateData(anchorset = anchors, normalization.method = "SCT", dims = 1:30)
saveRDS(integrated.data, paste0(data.dir, "integrated_tumors.rds"))