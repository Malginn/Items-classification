# Items's classification 
Classification of different product categories with an additional spam class.
Implemented with fine tuning of the base model EfficientNetV2M
1) download pre-collected data
2) in the Data class, image preprocessing is implemented
3) in the Classifier class, the neural network itself is implemented, which will be trained
we use multiclass categorical entropy, because we have 72 product categories + 1 spam category

Next, we integrate our model into production by loading it from the .h extension file, preprocessing the newly received data and making predictions on them.
Resulting Accuracy: 84%
