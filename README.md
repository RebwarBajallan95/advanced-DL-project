# MIMO - DD2412 Project #
This repository contains all code used to reproduce the results from the original MIMO paper and additional code for experiments with a MIMO U-net model. The U-net model is trained for the regression task of predicting the location of a receipt in low contrast images. The location is represented by 4 approximated corner points.

## Repository Structure ##
In the Unet directory all code used for training and generating results with the MIMO U-net model can be found. The root directory contains code used to reproduce the results in the original paper.

## Notes ##
* The training images and validation images for the U-net model is not available in this repository due to privacy concerns.
  * some example images can be found under /data/example_images/ and /data/example_predictions/