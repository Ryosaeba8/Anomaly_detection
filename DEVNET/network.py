# -*- coding: utf-8 -*-
import numpy as np
import warnings; warnings.simplefilter("ignore")


from sklearn.neighbors import KDTree
from sklearn.utils.random import sample_without_replacement



from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

MAX_INT = np.iinfo(np.int32).max

class Devnet_network :
    
    def __init__(self, hidden_dim=20, confidence_margin=5.0) :
        
        self.hidden_dim = hidden_dim
        self.confidence_margin = confidence_margin
        self.input_shape = None
    def deviation_loss(self, y_true, y_pred):
        '''
        z-score-based deviation loss
        '''      
        ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 5000) , dtype='float32')
        dev = (y_pred - K.mean(ref)) / K.std(ref)
        inlier_loss = K.abs(dev) 
        outlier_loss = K.abs(K.maximum(self.confidence_margin - dev, 0.))
        return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)
    
    def compile_model(self, input_shape) :
        self.input_shape = input_shape
        x_input = Input(shape=(self.input_shape, ))
        intermediate = Dense(self.hidden_dim, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
        intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
        self.model =  Model(x_input, intermediate)        

        rms = RMSprop(clipnorm=1.)
        self.model.compile(loss=self.deviation_loss, optimizer=rms)
        

class Trainer :
    
    def __init__(self, n_epochs=50, batch_size=256,
                 nb_batch=100, random_seed=42, path_model = "./model/"):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.rng = np.random.RandomState(random_seed)
        self.path_model = path_model
        
    def batch_generator_sup(self, x, outlier_indices, inlier_indices):
        """batch generator
        """
        rng = np.random.RandomState(self.rng.randint(MAX_INT, size = 1))
        counter = 0
        while 1:                
            ref, training_labels = self.input_batch_generation_sup(x, outlier_indices,
                                                              inlier_indices, rng)
            yield(ref, training_labels)
            if (counter > self.nb_batch):
                counter = 0
                
    def input_batch_generation_sup(self, x_train, outlier_indices, 
                                   inlier_indices, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''      
        dim = x_train.shape[1]
        ref = np.empty((self.batch_size, dim))    
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(self.batch_size):    
            if(i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = x_train[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = x_train[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels)                
                    
    def train(self, network, mode, x_train, 
              outlier_indices, inlier_indices,
              verbose=True) :           
        
        network.compile_model(x_train.shape[1])
        model_name = self.path_model + mode + "_"  + str(outlier_indices.shape[0]) + "_" +\
                    str(network.hidden_dim) + "_" + str(self.batch_size)
        
        checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                       save_best_only = True, save_weights_only = True)            
        
        network.model.fit_generator(self.batch_generator_sup(x_train, outlier_indices, 
                                                             inlier_indices),
                                    steps_per_epoch = self.nb_batch,
                                    epochs = self.n_epochs,
                                    callbacks=[checkpointer],
                                    verbose = verbose) 
        return network                  
                
                
class DevNet :
    def __init__(self, n_epochs=50, batch_size=256, n_neighbors=2,
                 nb_batch=100, random_seed=42, path_model = "./model/",
                 mode="semi_supervised", known_outliers=10, hidden_dim=20, 
                 confidence_margin=5.0, input_shape= 30, output=None, runs=None):
        assert (mode in ["semi_supervised", "unsupervised", "supervised"])
        self.mode = mode
        self.n_neighbors = n_neighbors       
        self.known_outliers = known_outliers
        self.Trainer = Trainer(n_epochs, batch_size, nb_batch, random_seed, path_model)
        self.network = Devnet_network(hidden_dim, confidence_margin)
        
    def prepare_data(self, x_train, y_train=None) :
        
        if self.mode == "unsupervised" :
            
            outlier_scores = self.lesinn(x_train, x_train) 
        
            ind_scores = np.argsort(outlier_scores.flatten())
            
            inlier_ids, outlier_ids = ind_scores[:-self.known_outliers:], ind_scores[-self.known_outliers:]  
            
        elif self.mode == "semi_supervised" :
            outlier_ids = np.where(y_train == 1)[0]
            
            if outlier_ids.shape[0] < self.known_outliers:
                outlier_scores = self.lesinn(x_train, x_train)
                ind_scores = np.argsort(outlier_scores.flatten())
                ind_scores = [elt for elt in ind_scores if elt not in outlier_ids]
                mn = self.known_outliers - outlier_ids.shape[0]
                to_add_idx = ind_scores[-mn:]
                
                outlier_ids = np.append(outlier_ids, to_add_idx)
            inlier_ids = np.delete(np.arange(len(x_train)), outlier_ids, axis=0)
        else :
            outlier_ids = np.where(y_train == 1)[0]
            inlier_ids = np.delete(np.arange(len(x_train)), 
                                   outlier_ids, axis=0)
            if outlier_ids.shape[0] > self.known_outliers:
                mn = outlier_ids.shape[0] -  self.known_outliers
                remove_idx = self.Trainer.rng.choice(outlier_ids, mn, replace=False)
                
                outlier_ids = np.array([elt for elt in outlier_ids if elt not in remove_idx]) ## to optimize
            
        self.inlier_ids = inlier_ids
        self.outlier_ids = outlier_ids
            
    def lesinn(self, x_train, to_query) :
        ensemble_size = 50
        subsample_size = 8
        scores = np.zeros([to_query.shape[0], 1])  
        seeds = self.Trainer.rng.randint(MAX_INT, size = ensemble_size)
        for i in range(0, ensemble_size):
            rs = np.random.RandomState(seeds[i])
            sid = sample_without_replacement(n_population = x_train.shape[0], 
                                             n_samples = subsample_size, 
                                             random_state = rs)
            subsample = x_train[sid]
            kdt = KDTree(subsample, metric='euclidean')
            dists, indices = kdt.query(to_query, k = self.n_neighbors)
            #import pdb; pdb.set_trace()
            dists = np.mean(dists, axis=1)[:, np.newaxis]
            scores += dists
        scores = scores / ensemble_size  
        return scores;        
    
    def fit(self, x_train, y_train=None, verbose=False) :
        
        self.prepare_data(x_train, y_train)
        self.network = self.Trainer.train(self.network, self.mode, x_train, 
                                          self.outlier_ids, self.inlier_ids,
                                          verbose=verbose) 
        
    def decision_function(self, x_val) :
        
        scores = self.network.model.predict(x_val)
        return scores
    
                
                
                
                
                
                
                