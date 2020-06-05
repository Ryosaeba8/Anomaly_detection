#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang

Source code for the REPEN algorithm in KDD'18. See the following paper for detail.
Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu. 2018. Learning Representations
of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. 
In KDD 2018: 24th ACM SIGKDD International Conferenceon Knowledge Discovery & 
Data Mining, August 19–23, 2018, London, UnitedKingdom.

This file is for experiments on csv data sets.
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np

MAX_INT = np.iinfo(np.int32).max
MAX_FLOAT = np.finfo(np.float32).max
#K.set_session(sess)

import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.utils.random import sample_without_replacement
import os
import sys
sys.path.append(os.path.dirname('../*'))
from utilities import  cutoff_unsorted, plot_precision_recall, \
  writeResults, Dataset, visualize_doc_embeddings
 
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Layer, Input
from keras.callbacks import ModelCheckpoint

def sqr_euclidean_dist(x,y):
    return K.sum(K.square(x - y), axis=-1);
 

class tripletRankingLossLayer(Layer):
    """Triplet ranking loss layer Class
    """

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(tripletRankingLossLayer, self).__init__(**kwargs)
        


    def rankingLoss(self, input_example, input_positive, input_negative):
        """Return the mean of the triplet ranking loss"""
        
        positive_distances = sqr_euclidean_dist(input_example, input_positive)
        negative_distances = sqr_euclidean_dist(input_example, input_negative)
        loss = K.mean(K.maximum(0., 1000. - (negative_distances - positive_distances) ))
        return loss
    
    def call(self, inputs):
        input_example = inputs[0]
        input_positive = inputs[1]
        input_negative = inputs[2]
        loss = self.rankingLoss(input_example, input_positive, input_negative)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return input_example; 
    
def lesinn(x_train, to_query):
    """the outlier scoring method, a bagging ensemble of Sp. See the following reference for detail.
    Pang, Guansong, Kai Ming Ting, and David Albrecht. 
    "LeSiNN: Detecting anomalies by identifying least similar nearest neighbours." 
    In Data Mining Workshop (ICDMW), 2015 IEEE International Conference on, pp. 623-630. IEEE, 2015.
    """
    rng = np.random.RandomState(42) 
    ensemble_size = 50
    subsample_size = 8
    scores = np.zeros([to_query.shape[0], 1])  
    # for reproductibility purpose  
    seeds = rng.randint(MAX_INT, size = ensemble_size)
    for i in range(0, ensemble_size):
        rs = np.random.RandomState(seeds[i])
        sid = sample_without_replacement(n_population = x_train.shape[0], 
                                         n_samples = subsample_size, 
                                         random_state = rs)
        subsample = x_train[sid]
        kdt = KDTree(subsample, metric='euclidean')
        dists, indices = kdt.query(to_query, k = 1)       
        scores += dists
    scores = scores / ensemble_size  
    return scores;


def batch_generator(X, labels, batch_size, steps_per_epoch, scores, rng,                           
                    positive_weights, negative_weights,
                    inlier_ids, outlier_ids):
    """batch generator
    """
    number_of_batches = steps_per_epoch
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:        
        X1, X2, X3 = tripletBatchGeneration(X, batch_size, rng, scores,
                           positive_weights, negative_weights,
                           inlier_ids, outlier_ids)
        counter += 1
        yield([np.array(X1), np.array(X2), np.array(X3)], None)
        if (counter > number_of_batches):
            counter = 0


def tripletBatchGeneration(X, batch_size, rng, outlier_scores,
                           positive_weights, negative_weights,
                           inlier_ids, outlier_ids):
    """batch generation
    """
    examples = np.zeros([batch_size]).astype('int')
    positives = np.zeros([batch_size]).astype('int')
    negatives = np.zeros([batch_size]).astype('int')   
    
    examples = rng.choice(inlier_ids, batch_size, p = positive_weights)
    positives = rng.choice(inlier_ids, batch_size)
    negatives = rng.choice(outlier_ids, batch_size, p = negative_weights)
#    for i in range(0, batch_size):
#        sid = rng.choice(len(inlier_ids), 1, p = positive_weights)
#        examples[i] = inlier_ids[sid]
#        
#        sid2 = rng.choice(len(inlier_ids), 1)
#        
#        while sid2 == sid:
#            sid2 = rng.choice(len(inlier_ids), 1)        
#        positives[i] = inlier_ids[sid2]
#        sid = rng.choice(len(outlier_ids), 1, p = negative_weights)
#        negatives[i] = outlier_ids[sid]
    examples = X[examples, :]
    positives = X[positives, :]
    negatives = X[negatives, :]
    return examples, positives, negatives;


    
def tripletModel(input_dim, embedding_size = 20): 
    """the learning model
    """

    input_e = Input(shape=(input_dim,), name = 'input_e')
    input_p = Input(shape=(input_dim,), name = 'input_p')
    input_n = Input(shape=(input_dim,), name = 'input_n')
    
    hidden_layer = Dense(embedding_size, activation='relu', name = 'hidden_layer')
    hidden_e = hidden_layer(input_e)
    hidden_p = hidden_layer(input_p)
    hidden_n = hidden_layer(input_n)
    
    output_layer = tripletRankingLossLayer()([hidden_e,hidden_p,hidden_n])
    
    rankModel = Model(inputs=[input_e, input_p, input_n], outputs=output_layer)
    
    representation = Model(inputs=input_e, outputs=hidden_e)
    return rankModel, representation;

    
def training_model(rankModel, X, labels,embedding_size, scores,
                   filename, ite_num, rng,
                   positive_weights, negative_weights,
                   inlier_ids, outlier_ids, nb_epoch=30):  
    """training the model
    """
    
    rankModel.compile(optimizer = 'adadelta', 
                      loss = None)
    
    checkpointer = ModelCheckpoint("./model/" + str(embedding_size) + "D_" + str(ite_num) + "_"+ filename + ".h5", monitor='loss',
                               verbose=0, save_best_only = True, save_weights_only=True)
    
    batch_size = 128    
    samples_per_epoch = X.shape[0]
    steps_per_epoch = samples_per_epoch / batch_size
    rankModel.fit_generator(batch_generator(X, labels, batch_size, steps_per_epoch, scores, rng,
                                                       positive_weights, negative_weights,
                                                       inlier_ids, outlier_ids),
                              steps_per_epoch = steps_per_epoch,
                              epochs = nb_epoch,
                              shuffle = False,
                              callbacks=[checkpointer],
                              verbose = False);


def load_model_predict(model_name, dataset, labels, embedding_size, filename):
    
    """load the representation learning model and do the mappings.
    LeSiNN, the Sp ensemble, is applied to perform outlier scoring
    in the representation space.
    """
    rankModel, representation = tripletModel(X.shape[1], embedding_size=embedding_size)  
    rankModel.load_weights(model_name)
    representation = Model(inputs=rankModel.input[0],
                                 outputs=rankModel.get_layer('hidden_layer').get_output_at(0))
    
    X_train_rep = representation.predict(dataset.X_train)
    X_val_rep = representation.predict(dataset.X_val)
    
    outlier_scores_tr = lesinn(X_train_rep, X_train_rep) 
    outlier_scores_val = lesinn(X_train_rep, X_val_rep) 
    
    rauc = plot_precision_recall([outlier_scores_tr, outlier_scores_val],
                          [dataset.Y_train, dataset.Y_val],
                          ['train', 'Val'], name_model= model_name.split('/')[-1].split(".")[0],
                          mode="min_max")
    
    visualize_doc_embeddings(X_val_rep,
                          ['blue','red',], 
                          dataset.Y_val, model_name.split('/')[-1].split(".")[0],
                          center = X_val_rep.mean(axis=0))
    return rauc

def test_diff_embeddings(dataset, labels, outlier_scores, filename,
                           positive_weights, negative_weights,
                           inlier_ids, outlier_ids):
    """sensitivity test w.r.t. different representation dimensions
    """
    embeddings = np.array([2, 3, 5, 10])
    for j in range(0, len(embeddings)):
        embedding_size = embeddings[j]
        test_single_embedding(dataset, labels, outlier_scores, filename, embedding_size,
                           positive_weights, negative_weights,
                           inlier_ids, outlier_ids)
        

def test_single_embedding(dataset, labels, outlier_scores, filename, 
                          embedding_size,
                           positive_weights, negative_weights,
                           inlier_ids, outlier_ids):
    """perform representation learning with a fixed representation dimension
    and outlier detection using LeSiNN
    """
    runs = 3
    rauc = np.empty([runs, 1])    
    rng = np.random.RandomState(42) 
    print(" ")
    print("Dimension : " + str(embedding_size) )
    for i in range(0, runs):
        rankModel, representation = tripletModel(X.shape[1], embedding_size)    
        training_model(rankModel, dataset.X_train, labels, embedding_size, outlier_scores, filename, i, rng,
                           positive_weights, negative_weights,
                           inlier_ids, outlier_ids)
        
        modelName = "./model/" + str(embedding_size) + "D_" + str(i)+ "_" + filename + '.h5'
        
        rauc[i] = load_model_predict(modelName, dataset, labels, embedding_size, filename)
    mean_auc = np.mean(rauc)
    s_auc = np.std(rauc)
    writeResults(filename, embedding_size, mean_auc, std_auc = s_auc)

    
debug = False
if __name__ =='__main__' :    
    
    print()
    dataset = Dataset()
    ## specify data files    
    X, labels = dataset.X_train, dataset.Y_train
    
    #start_time = time.time() 
    outlier_scores_train = lesinn(X, X) 
    outlier_scores_val = lesinn(X,  dataset.X_val) 
    
    plt.ioff()        
    plot_precision_recall([outlier_scores_train, outlier_scores_val],
                          [dataset.Y_train, dataset.Y_val],
                          ['train', 'Val'], name_model= 'LeSiNN',
                          mode ='min_max')
    
    embedding_size = 2
    rankModel, representation = tripletModel(X.shape[1], embedding_size) 
    rng = np.random.RandomState(42) 
    
    mode = 'supervised'
    
    if mode == "unsupervised" :
        
        inlier_ids, outlier_ids = cutoff_unsorted(outlier_scores_train)
        transforms = np.sum(outlier_scores_train[inlier_ids]) - outlier_scores_train[inlier_ids]
        total_weights_p = np.sum(transforms)
        
        positive_weights = transforms / total_weights_p
        positive_weights = positive_weights.flatten()
        total_weights_n = np.sum(outlier_scores_train[outlier_ids])
        negative_weights = outlier_scores_train[outlier_ids] / total_weights_n
        negative_weights = negative_weights.flatten()
        
    else :
         inlier_ids, outlier_ids = np.where(labels == 1)[0], np.where(labels == -1)[0]
         positive_weights = np.ones(inlier_ids.shape[0]) * (1/inlier_ids.shape[0])
         negative_weights = np.ones(outlier_ids.shape[0]) * (1/outlier_ids.shape[0]) 
         

    if debug == True :        
        training_model(rankModel, X, labels,
                       embedding_size, outlier_scores_train,
                       'model', 1, rng,
                       positive_weights, negative_weights,
                       inlier_ids, outlier_ids, nb_epoch=100)
        
        
        rankModel, representation = tripletModel(X.shape[1], embedding_size=embedding_size)  
        rankModel.load_weights('model/8D_1_model.h5')
        representation = Model(inputs=rankModel.input[0],
                               outputs=rankModel.get_layer('hidden_layer').get_output_at(0))
        
        X_train_rep = representation.predict(dataset.X_train)
        X_val_rep = representation.predict(dataset.X_val)
        
        outlier_scores_tr = lesinn(X_train_rep, X_train_rep) 
        outlier_scores_val = lesinn(X_train_rep, X_val_rep) 
        
        plot_precision_recall([outlier_scores_tr, outlier_scores_val],
                              [dataset.Y_train, dataset.Y_val],
                              ['train', 'Val'], name_model= 'REPEN_dim_8',
                              mode="min_max")
        
        visualize_doc_embeddings(X_val_rep,
                              ['blue','red',], 
                              dataset.Y_val, "REPEN_dim_8",
                              center = X_val_rep.mean(axis=0))
    else :       
        
        filename = mode
        test_diff_embeddings(dataset, labels, outlier_scores_train, filename,
                                   positive_weights, negative_weights,
                                   inlier_ids, outlier_ids)

#test_single_embedding(X, labels, outlier_scores, filename)
#writeResults(filename, X.shape[1], rauc)
#outlier_scores = None
#test_single_embedding(X, labels, outlier_scores, filename)
#print("--- %s seconds ---" % (time.time() - start_time))
#writeOutlierScores(outlier_scores, labels, filename)
#test_diff_embeddings(X, labels, outlier_scores, filename)


