# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

import numpy as np
import warnings; warnings.simplefilter("ignore")
np.random.seed(42)

import argparse
import numpy as np
import sys
from sklearn.neighbors import KDTree
from sklearn.utils.random import sample_without_replacement
#from scipy.sparse import vstack, csc_matrix
from utils import writeResults, aucPerformance, cutoff_unsorted
import os
sys.path.append(os.path.dirname('../*'))

from utilities import Dataset, visualize_doc_embeddings
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint


import time

MAX_INT = np.iinfo(np.int32).max
data_format = 0


def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)


def deviation_loss(y_true, y_pred):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 2.     
    ## size=5000 is the setting of l in algorithm 1 in the paper
    ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 5000) , dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss = K.abs(dev) 
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def deviation_network(input_shape):
    '''
    construct the deviation network-based detection model
    '''
    model = dev_network_s(input_shape)
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        ref, training_labels = input_batch_generation_sup(x, outlier_indices,
                                                          inlier_indices, 
                                                          batch_size, rng)
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0
 
def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

def load_model_weight_predict(model_name, input_shape, network_depth, x_test):
    '''
    load the saved weights to make predictions
    '''
    model = deviation_network(input_shape, network_depth)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)    
    
    scores = scoring_network.predict(x_test)
    return scores

def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''  
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

def lesinn(x_train, to_query, n_neighbor=1):
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
        dists, indices = kdt.query(to_query, k = n_neighbor)       
        scores += dists
    scores = scores / ensemble_size  
    return scores;

def run_devnet(args):
    print("Chosen mode :", args.mode)
    nm = 'fraud'
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    
    runs = args.runs
    rauc = np.zeros(runs)
    ap = np.zeros(runs)  
    filename = nm.strip()
    global data_format
    data_format = int(args.data_format)
    
    data = Dataset(mode="other")
    
    if args.mode =="unsupervised" :
        outlier_scores = lesinn(data.X_train, data.X_train) 
        
        ind_scores = np.argsort(outlier_scores.flatten())
            
        inlier_ids, outlier_ids = ind_scores[:-args.known_outliers:], ind_scores[-args.known_outliers:]  
        inlier_ids = np.intersect1d(inlier_ids, np.where(data.Y_train == 0)[0])

    #print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], 
    #                                                        n_outliers))    
    train_time = 0
    test_time = 0
    for i in np.arange(runs):  
        print(filename + ': round ' + str(i))     
        x_train, x_test, y_train, y_test = data.X_train, data.X_val, data.Y_train, data.Y_val
        
        
        if args.mode == "unsupervised" :
            y_train[inlier_ids] = 0;
            y_train[outlier_ids] = 1
        
        outlier_indices = np.where(y_train == 1)[0]
        outliers = x_train[outlier_indices]  
        n_outliers_org = outliers.shape[0]   
        
        inlier_indices = np.where(y_train == 0)[0]
        n_outliers = len(outlier_indices)        
        
        n_noise  = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
        n_noise = int(n_noise)                
        
        rng = np.random.RandomState(random_seed)  
        if data_format == 0:                
            if n_outliers > args.known_outliers:
                mn = n_outliers - args.known_outliers
                remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                x_train = np.delete(x_train, remove_idx, axis=0)
                y_train = np.delete(y_train, remove_idx, axis=0)
        if args.cont_rate > 0 :
            noises = inject_noise(outliers, n_noise, random_seed)
            x_train = np.append(x_train, noises, axis = 0)
            y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
        
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]
        #print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
        input_shape = x_train.shape[1:]
        n_samples_trn = x_train.shape[0]
        n_outliers = len(outlier_indices)            
        print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
        
        
        start_time = time.time() 
        input_shape = x_train.shape[1:]
        epochs = args.epochs
        batch_size = args.batch_size    
        nb_batch = args.nb_batch  
        model = deviation_network(input_shape)
        #print(model.summary())  
        model_name = "./model/" + args.mode + "_" + str(args.cont_rate) + "cr_"  + str(args.known_outliers)  +"d.h5"
        checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                       save_best_only = True, save_weights_only = True)            
        
        model.fit_generator(batch_generator_sup(x_train, outlier_indices, 
                                                inlier_indices, batch_size, nb_batch, rng),
                                      steps_per_epoch = nb_batch,
                                      epochs = epochs,
                                      callbacks=[checkpointer],
                                      verbose = True)   
        train_time += time.time() - start_time
        
        start_time = time.time() 
        scores = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
        test_time += time.time() - start_time
        rauc[i], ap[i] = aucPerformance(scores, y_test)     
    
    mean_auc = np.mean(rauc)
    #std_auc = np.std(rauc)
    mean_aucpr = np.mean(ap)
    std_aucpr = np.std(ap)
    train_time = train_time/runs
    test_time = test_time/runs
    print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
    #print("average runtime: %.4f seconds" % (train_time + test_time))
    writeResults(filename+'_vrai_'+str(network_depth), n_samples_trn, n_outliers_org, n_outliers,
                 mean_aucpr, std_aucpr, args.cont_rate, path=args.output)

def plot_pred(name_model, data) :
    #data = Dataset(mode="other")
    x_test, y_test = data.X_val, data.Y_val
    input_shape = x_test.shape[1:]
    network_depth = int(args.network_depth)
    model = deviation_network(input_shape, network_depth)   
    model.load_weights(name_model)
    intermediate_layer = Model(inputs = model.input, outputs = model.get_layer('hl1').output)
    encode = intermediate_layer.predict(x_test)
    visualize_doc_embeddings(encode, ['blue','red'], y_test, name_model.split("/")[1], None)    
      
parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1','2', '4'], default='2', help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=256, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=100, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=5, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=10, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0, help="the outlier contamination rate in the training data")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--data_set", type=str, default='fraud', help="a list of data set names")
parser.add_argument("--data_format", choices=['0','1'], default='0',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str, default='./results/performance.csv', help="the output file path")
parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
parser.add_argument("--mode", type=str, default="unsupervised", help="the mode of the algorithm")

args = parser.parse_args()
for n_outliers in [50, 100, 500, 2000, 5000, 10000] :
    args.known_outliers = n_outliers
    run_devnet(args)


import glob; all_names = glob.glob("model/*")
data = Dataset(mode="other")
for name_model in all_names :
    try :
        print(name_model); plot_pred(name_model, data)
    except :
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    