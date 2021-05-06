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
            

class flexi_net(Sequential):    
    
    def __init__(self, input_vector_size=2048, output_size=(256,1024), reg=0.0001, learning_rate=1e-3): 
        
        out_0 = output_size[0]
        out_1 = output_size[1]
        
        if out_0==out_1: 
            f0=f1=4
        else:
            f0=2
            f1=8
        
        super(flexi_net, self).__init__(layers=None, name=None)
        
        assert input_vector_size>=2048, "Flexi_nn: Too few input data!"
        
        self.add(Dense(input_vector_size, input_shape=(input_vector_size,), activation=None))
        
        self.add(Reshape((f0,f1,int(input_vector_size/16)),input_shape=(input_vector_size,))) 
        
        self.add(UpSampling2D())
        self.add(Conv2DTranspose(int(input_vector_size/32),(3,3), strides=1, padding='same', activation='hard_sigmoid'))          
        self.add(BatchNormalization())
        
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(input_vector_size/64),(3,3), strides=1, padding='same', activation='relu'))          
        self.add(BatchNormalization())
        
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(input_vector_size/128),(3,3), strides=1, padding='same', activation='relu'))          
        self.add(BatchNormalization())
                
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(input_vector_size/256),(3,3), strides=1, padding='same', activation='relu'))          
        self.add(BatchNormalization())
        
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(input_vector_size/512),(3,3), strides=1, padding='same', activation='relu'))          
        self.add(BatchNormalization())
        
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(input_vector_size/1024),(3,3), strides=1, padding='same', activation='relu'))          
        self.add(BatchNormalization())
        
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(input_vector_size/2048),(3,3), strides=1, padding='same', activation='relu'))          
        self.add(BatchNormalization())
        
        self.add(Conv2D(1, (3,3), activation='hard_sigmoid', padding='same'))                                                        

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse'])

        print('Light parametrization net output shape:', self.layers[len(self.layers)-1].output_shape)
        
        

class light_param_net(Sequential):    
    
    def __init__(self, num_coefs=400, add_coef_layer=False, reg=0.0001, learning_rate=1e-3):        
        super(light_param_net, self).__init__(layers=None, name=None)
                
        if add_coef_layer:
            self.add(Dense(num_coefs, input_shape=(1,), use_bias=False, kernel_regularizer=l2(reg)))

        self.add(Dense(num_coefs, input_shape=(num_coefs,), activation=None))
        self.add(Reshape((2,2,int(num_coefs/4)),input_shape=(num_coefs,)))                             
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/8),(3,3), strides=1, activation='hard_sigmoid'))          
        self.add(BatchNormalization())
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/16),(3,3), strides=1, activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/32),(3,3), strides=1, activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/64),(3,3), strides=1, activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/128),(3,3), strides=1, activation='relu'))        
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/256),(3,3), strides=1, activation='relu'))        
        self.add(BatchNormalization())                                                               
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(1,(3,3), strides=1, activation='relu'))                       
        self.add(BatchNormalization())                                                         
        self.add(Conv2D(1, (3,3), activation='hard_sigmoid'))                                                        
        self.add(ZeroPadding2D((2,2)))                                 

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse'])

        print('Light parametrization net output shape:', self.layers[len(self.layers)-1].output_shape)

class heavy_param_net(Sequential):    
    
    def __init__(self, num_coefs=400, add_coef_layer=False, reg=0.0001, learning_rate=1e-3):        
        super(heavy_param_net, self).__init__(layers=None, name=None)
        
        if add_coef_layer:
            self.add(Dense(num_coefs, input_shape=(1,), use_bias=False, kernel_regularizer=l2(reg)))  
        
        self.add(Dense(2048, input_shape=(num_coefs,), activation='tanh'))#hard_sigmoid
        self.add(Reshape((2,2,512)))                             
        self.add(UpSampling2D())
        self.add(Conv2D(256,(2,2), strides=1, padding='same')) 
        self.add(Conv2DTranspose(256,(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2DTranspose(256,(3,3), strides=1, padding='same', activation='relu')) 
        self.add(BatchNormalization())
        self.add(UpSampling2D())   
        self.add(Conv2D(128,(2,2), strides=1, padding='same')) 
        self.add(Conv2DTranspose(128,(3,3), strides=1, padding='same', activation='relu'))  
        self.add(Conv2DTranspose(128,(3,3), strides=1, padding='same', activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())     
        self.add(Conv2D(64,(2,2), padding='same', strides=1)) 
        self.add(Conv2DTranspose(64,(3,3), strides=1, padding='same', activation='relu')) 
        self.add(Conv2DTranspose(64,(3,3), strides=1, padding='same', activation='relu')) 
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())  
        self.add(Conv2D(32,(2,2), padding='same', strides=1))
        self.add(Conv2DTranspose(32,(3,3), strides=1, padding='same', activation='relu'))  
        self.add(Conv2DTranspose(32,(3,3), strides=1, padding='same', activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D()) 
        self.add(Conv2D(16,(2,2), padding='same', strides=1))
        self.add(Conv2DTranspose(16,(3,3), strides=1, padding='same', activation='relu'))    
        self.add(Conv2DTranspose(16,(3,3), strides=1, padding='same', activation='relu'))
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())   
        self.add(Conv2D(8,(2,2), padding='same', strides=1))
        self.add(Conv2DTranspose(8,(3,3), strides=1, padding='same', activation='relu'))   
        self.add(Conv2DTranspose(8,(3,3), strides=1, padding='same', activation='relu')) 
        self.add(BatchNormalization())                                                               
        self.add(UpSampling2D())  
        self.add(Conv2D(4,(2,2), padding='same', strides=1))
        self.add(Conv2DTranspose(4,(3,3), strides=1, padding='same', activation='relu'))    
        self.add(Conv2DTranspose(4,(3,3), strides=1, padding='same', activation='relu')) 
        self.add(BatchNormalization())   
        self.add(UpSampling2D())  
        self.add(Conv2D(4,(2,2), padding='same', strides=1))
        self.add(Conv2DTranspose(4,(3,3), strides=1, padding='same', activation='relu'))    
        self.add(Conv2DTranspose(4,(3,3), strides=1, padding='same', activation='relu')) 
        self.add(BatchNormalization())  
        self.add(Conv2D(1, (3,3), padding='same', activation='hard_sigmoid'))                                                 

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse'])
        
        print('Heavy parametrization net output shape:', self.layers[len(self.layers)-1].output_shape)
        
        
class ldm_net(Sequential):    
    
    def __init__(self, num_coefs=3072, reg=0.0001, learning_rate=1e-3, loss='mse', metrics=['mse'], verbose=False):   
        
        super(ldm_net, self).__init__(layers=None, name=None)
                
        self.add(Dense(num_coefs, input_shape=(num_coefs,), activation=None))
        self.add(Reshape((2,2,int(num_coefs/4)),input_shape=(num_coefs,)))                             
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/8),(3,3), strides=1, activation='hard_sigmoid'))          
        self.add(BatchNormalization())
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/16),(3,3), strides=1, activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/32),(3,3), strides=1, activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/64),(3,3), strides=1, activation='relu'))         
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/128),(3,3), strides=1, activation='relu'))        
        self.add(BatchNormalization())                                                         
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(int(num_coefs/256),(3,3), strides=1, activation='relu'))        
        self.add(BatchNormalization())                                                               
        self.add(UpSampling2D())                                                               
        self.add(Conv2DTranspose(1,(3,3), strides=1, activation='relu'))                       
        self.add(BatchNormalization())                                                         
        self.add(Conv2D(1, (3,3), activation='hard_sigmoid'))                                                        
        self.add(ZeroPadding2D((2,2)))                                 

        self.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)

        if verbose: print('LDM net output shape:', self.layers[len(self.layers)-1].output_shape)
