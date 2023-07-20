# Ventriculomegaly-MEng
All code involved in the Ventriculomegaly ML Classification project 

There are two main parts to this project, the models and the segmentation.

The segmentation scripts take a predetermined parcellation/segmentation and apply it to a raw metric in order to produce a set of features that can be used to train a ML model on. These are outputted in the form of a csv file. A parcellation is needed for this to function as well as the metric files.

The models take the previously mentioned features and use them to train a Gradient Boosted Classifier on each individual metric. These predictions are then concatenated together and used to train a meta model on the predictions of the previous component models. A train test validation split is used to prevent data leaking yet still allow the meta model the ability to 'percieve' how well the models are on unseen data.

The final folder is used for the calculation of cortical surface area as it isnt normally automatically calculated.

This will be further edited and improved
