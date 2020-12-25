from tensorflow.keras.utils import Sequence
import numpy as np

class BatchGenerator(Sequence):
    
    def __init__(self, X_input, y_input, list_IDs, batch_size=4, Xdim=(64,64,1), ydim=(128,), shuffle=True, dtype_x='float64', dtype_y='int8'):
        
        self.Xdim = Xdim
        self.ydim = ydim
        self.batch_size = batch_size        
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.X_input = X_input
        self.y_input = y_input
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
       
        X = np.empty((self.batch_size, *self.Xdim), dtype = self.dtype_x)
        y = np.empty((self.batch_size, *self.ydim), dtype = self.dtype_y)
        
        for i, ID in enumerate(list_IDs_temp):            
            X[i,] = self.X_input[ID,]
            y[i,] = self.y_input[ID,]

        return X, y
    
    
    