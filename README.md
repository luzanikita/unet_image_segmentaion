# unet_image_segmentaion
A semantic segmentation model with UNet architecture using Keras
___

Kaggle competion: https://www.kaggle.com/c/data-science-bowl-2018
___

Task was complited with help of https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

### Plan:

1. Importing modules for forking with arrays and dataframes (numpy, pandas), images (skimage), building deeplearning model (keras, tensorflow)
2. Getting input data from folders
3. Preprocess pictures (merging masks, resizing images)
4. Establishing metric function - dice coef
5. Building u-net model
6. Training model and saving weights (validation split - 0.1 data which are using to calculate metrics on untrained data, 15 epochs are enough to not be overfitted (loss score stops growing))
7. Loading model's weights from file and make prediction on train, test data
8. Converting prediction values to [0, 1] (threshold = 0.5)
9. Showing random samples from test dataset (picture - mask)
___

### Instructions:

- run Train.py to get train data, preprocess it, build model and train it
    output: logs in console (training process, metrics), saved weights in .h5 file

- run Preduct_masks.py to get test data, preprocess it, load model, make predictions and view 5 pairs (test picture - predicted mask). We can't calcultate metric on test data, because we don't have original masks to compare with

- model.ipynb is jupyter notebook with building model, testing it, viewing results and performance (during the test was detected class of images which has very low dice coef)
___

Details about modules in requirements.txt