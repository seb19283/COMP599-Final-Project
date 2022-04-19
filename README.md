# COMP599-Final-Project

Data files should be downloaded from the following Kaggle dataset and added directly into the ClusteredSBERT/Data/ directory. https://www.kaggle.com/datasets/sebconn/nqretrievertraining

This code is also available on a Kaggle notebook: https://www.kaggle.com/code/rodrigolisboamirco/final-project

In order to run the experiments from the paper, execute evaluate_models.py and train_models.py passing the name of the dataset that you want to evaluate/train on as the only argument. The options are 'nq', 'webq' and 'triviaqa'.

evaluate_models.py will evaluate all 3 models (ClusteredSBERT, BruteforceSBERT and BruteforceDPR) on the given dataset.

train_models.py will evaluate ClusteredSBERT without any fine-tuning. Then it will train the model retriever with the technique exposed in the paper. After the training is done, it will first recompute all document embeddings and rearrange them under the existing clusters. Another evaluation step is done at this point. Finally, a new clustering is done from scratch, and the model is evaluated once again.