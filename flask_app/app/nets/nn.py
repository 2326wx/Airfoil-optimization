from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose, Conv2D, ZeroPadding2D, Softmax, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class nn_2561024(Sequential):
    def __init__(self, num_coefs=3072, reg=0.0001, learning_rate=1e-3, loss='mse', metrics=['mse'], verbose=False):           
        super(nn_2561024, self).__init__(layers=None, name=None)                
        self.add(Dense(num_coefs, input_shape=(num_coefs,), activation=None))
        self.add(Reshape((2,8,int(num_coefs/16)),input_shape=(num_coefs,)))                             
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/32),(3,3), strides=1, padding='same', activation='hard_sigmoid'))          
        self.add(BatchNormalization())
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/64),(3,3), strides=1, padding='same', activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/128),(3,3), strides=1, padding='same',activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/256),(3,3), strides=1, padding='same', activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/512),(3,3), strides=1, padding='same', activation='relu'))        
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/1024),(3,3), strides=1, padding='same', activation='relu'))        
        self.add(BatchNormalization())                                                               
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(1,(3,3), strides=1, padding='same', activation='relu'))                       
        self.add(BatchNormalization())                                                         
        self.add(Conv2D(1, (3,3), padding='same', activation='hard_sigmoid'))
        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
        if verbose: print('NN output shape:', self.layers[len(self.layers)-1].output_shape)
            

