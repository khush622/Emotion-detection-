import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

dr = 0.6
X_input =  Input(X_train.shape[1:])
X       =  Dropout(.2)(X_input)
X       =  Conv2D(50,(5,5),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X_input)
X       =  BatchNormalization()(X)
X       =  Conv2D(50,(5,5),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  MaxPooling2D(2,2)(X)
X       =  BatchNormalization()(X)
X       =  Dropout(dr)(X)
X       =  Conv2D(100,(5,5),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X1      =  Conv2D(100,(7,7),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  BatchNormalization()(X)
X       =  Conv2D(100,(7,7),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  MaxPooling2D(2,2)((X+X1)/2)
X       =  BatchNormalization()(X)
X       =  Dropout(dr)(X)
X       =  Conv2D(200,(7,7),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  BatchNormalization()(X)
X       =  Conv2D(200,(3,3),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  MaxPooling2D(2,2)(X)
X       =  BatchNormalization()(X)
X       =  Dropout(dr)(X)
X       =  Conv2D(400,(3,3),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  BatchNormalization()(X)
X       =  Conv2D(400,(3,3),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  MaxPooling2D(2,2)(X)
X       =  BatchNormalization()(X)
X       =  Dropout(dr)(X)
X       =  Conv2D(800,(2,2),activation = 'relu',kernel_initializer=tf.keras.initializers.HeNormal())(X)
X       =  BatchNormalization()(X)
X       =  Flatten()(X)
X       =  Dense(200,activation = 'relu')(X)
X       =  Dense(6,activation = 'softmax')(X)

model   =  Model(inputs = X_input, outputs = X)

model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
