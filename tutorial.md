## Instructions for use ##

1. For bug report and a complete demonstration, please visit: https://github.com/gevaertlab/GBM360 
2. Click the `Run` tab located at the top of the page.
3. To start the analysis, user can either upload a new histology image or simply click `Use an example slide`. <br>
    **Note**: 
    - We currently support images saved in *tif*, *tiff* or *svs* format. <br>
    - Ideally, the image should be scanned at 20X magnification with a pixel resolution of 0.5um / pixel.

A thumbnail of the image will display when the upload is completed

4. Select the mode for running the job. <br>
    **Note**: 
    
    - The default mode is set to the `Test mode`, which will only predicts a limited portion of the image (1,000 patches). This is meant to speed up the process by generating a quick preview of the results. 
    - To predict the entire image, please switch to `Complete` mode.
    - We are currently working on obtaining GPU support for this software, which will significantly accelerate its performance.
    
5. Click the `Get cell type visualization` button to predict the spatial distribution of transcriptional subtype for tumor cells.


The image will be colored by the predicted transcriptional subtype:


6. Based on the spatial subtype prediction, the software will automatically make several statistical analysis to quantify subtype compositions and spatial cellular organization:

    (1) Subtype fraction

    (2) Subtype interaction

    (3) Clustering coefficient 

6. Finally, click the `Get prognosis visualization` button to predict the aggressive scores. 

Blue indicates low aggressiveness, while Red indicates high aggressiveness
