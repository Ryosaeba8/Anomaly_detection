import warnings; warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt     
import numpy as np
layers = tf.keras.layers

class Network :
    def __init__(self, network_type, n_units=32, kernel_size=5,
                 dropout=0.1, future_step=1, pool_size=5) :
        model = Sequential()
        if network_type == "lstm" :
            
            model.add(layers.LSTM(n_units, return_sequences=True))
            if dropout !=None :
                model.add(layers.Dropout(dropout))
            model.add(layers.LSTM(n_units, return_sequences=False))
            if dropout !=None :
                model.add(layers.Dropout(dropout))
            model.add(layers.Dense(future_step))
            
        elif network_type == "cnn" :
            
            model.add(layers.Conv1D(filters=n_units, kernel_size=kernel_size,
                                    activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=pool_size))
            model.add(layers.Conv1D(filters=n_units, kernel_size=kernel_size,
                                    activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=pool_size))
            model.add(layers.Flatten())
            model.add(layers.Dense(future_step))
        elif network_type == "ae" :
            print("pas encore implementé")
            
        self.model = model
    def compile_model(self, loss="mean_absolute_error", metrics=None):
        self.model.compile(loss=loss,
                           optimizer="adam",
                           metrics=metrics)
        return self.model
        
        
class TSPredictor :
    def __init__(self, network_type="lstm", 
                 n_units=32, dropout=0.1, pool_size=2, kernel_size=5,
                 future_step=1, epochs=20, batch_size=128) :
        self.network_type = network_type
        self.model = Network(network_type=network_type, n_units=n_units, 
                             dropout=dropout, future_step=future_step,
                             pool_size=pool_size, kernel_size=kernel_size).compile_model()
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self, X_train, Y_train,
            X_val=None, Y_val=None, verbose=True) :
        
        if type(X_val) == type(None) :
            history = self.model.fit(X_train, Y_train, epochs=self.epochs,
                                     batch_size=self.batch_size, 
                                     verbose=verbose, shuffle=True)            

        else :
            history = self.model.fit(X_train, Y_train, epochs=self.epochs,
                                     batch_size=self.batch_size, 
                                     verbose=verbose, shuffle=True,
                                     validation_data=(X_val,Y_val))
            
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(self.network_type + ' loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()
        plt.savefig(self.network_type + '_loss.png')
        
    def predict(self, X_val) :
        y_pred = self.model.predict(X_val)
        return y_pred
    
    def decision_function(self, X_val, y_val) :
        scores = np.linalg.norm(self.predict(X_val) - y_val, axis=1)
        return scores














































