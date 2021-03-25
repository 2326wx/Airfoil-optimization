import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
import numpy as np

class BatchGenerator(Sequence):
    
    def __init__(self, X_input, y_input, list_IDs, batch_size=4, Xdim=(64,64,1), ydim=(128,), shuffle=True, dtype_x='float64', dtype_y='float64'):
        
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
    
    
def tversky_loss(y_true, y_pred):
    
    #https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
    
    y_true = K.cast(y_true,'float32')
    
    alpha = 0.50
    beta  = 0.50
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def MSE_var(y_true, y_pred):
    if y_true.shape[0]:
        vol=1
        for d in y_true.shape: vol=vol*d    
        y_true = K.reshape(y_true, (y_true.shape[0], int(vol/y_true.shape[0])))
        y_pred = K.reshape(y_pred, (y_true.shape[0], int(vol/y_pred.shape[0])))
        MSE = K.sum(K.pow(K.abs(y_true-y_pred), 2)/y_true.shape[1],axis=1)
        return float(K.var(MSE))
    else:
        return -1
    
    
def IoU_var(y_true, y_pred):
        
    if y_true.shape[0]:
        vol=1
        for d in y_true.shape: vol=vol*d    
        y_true = K.reshape(y_true, (y_true.shape[0], int(vol/y_true.shape[0])))
        y_pred = K.reshape(y_pred, (y_pred.shape[0], int(vol/y_pred.shape[0]))) 
        
        #if np.sum(y_true)==0: print(' Sum(true)=0!')
        #if np.sum(y_pred)==0: print(' Sum(pred)=0!')        
        
        y_true=np.array(y_true).astype('bool')
        y_pred=np.array(y_pred)        
        y_pred[y_pred<0.5]=False
        y_pred[y_pred>=0.5]=True   
        y_pred[y_pred==np.nan]=False
        y_true[y_true==np.nan]=False  
        y_pred=np.array(y_pred).astype('bool')
        
        intersection = y_true*y_pred
        union = y_true + y_pred

        try:
            nom = np.sum(intersection,axis=1)
            denom = np.sum(union,axis=1)             
            if 0 in denom: 
                denom[denom==0]=np.mean(denom)
                nom[denom==0]=np.mean(nom)            
            iou_score = nom / denom
            return float(np.var(iou_score))
        except:
            return -1        
    else:
        return 0
    