import tensorflow as tf
import pandas as pd
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier 
from time import time
import time
from sklearn.model_selection import GridSearchCV

#Reading MNIST data

def readData():
  def load_mnist(path, kind='train'):
      """Load MNIST data from `path`"""
      labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte' % kind)
      images_path = os.path.join(path,
                                '%s-images-idx3-ubyte' % kind)
      with open(labels_path, 'rb') as lbpath:
          magic, n = struct.unpack('>II',
                                  lbpath.read(8))
          labels = np.fromfile(lbpath,
                              dtype=np.uint8)
      with open(images_path, 'rb') as imgpath:
          magic, num, rows, cols = struct.unpack(">IIII",
                                                imgpath.read(16))
          images = np.fromfile(imgpath,
                              dtype=np.uint8).reshape(
                              len(labels), 784)
          images = ((images / 255.) - .5) * 2
      return images, labels

  ## loading the data
  X_train, y_train = load_mnist('/home/est1/Edgar/tensorflowPractice/mnist', kind='train')
  #print('Rows: %d, Columns: %d' %(X_train.shape[0],
  #                                X_train.shape[1]))

  X_test, y_test = load_mnist('/home/est1/Edgar/tensorflowPractice/mnist', kind='t10k')
  #print('Rows: %d, Columns: %d' %(X_test.shape[0],
  #                                X_test.shape[1]))

  # With the function above the X_train is normalized in -1 to 1.
  # This normalization is made because the gradient-based optimization is much more stable.

  # Visualize the normalized data.
  #plt.imshow(X_train[0].reshape((28,28)))

  # Becoming 2D (X_train) to 3D(X_train)
  X_train = X_train[:,:,np.newaxis]
  X_test = X_test[:,:,np.newaxis]

  # Reshaping d(60k,784,1) to d(60k,28,28)
  X_train = X_train.reshape((len(y_train),28,28))
  X_test = X_test.reshape((len(y_test),28,28))
  return X_train, X_test, y_train, y_test

def rowColumnInputShape(X_train,X_test,y_train, y_test,rRow=1,rFlag=0,rLayer=1,rUnits=5,formReadPixel ='H'):

  for row in range(rRow,(784+1),1):
    if((784%row)==0):
      column = int(784/row)
      #print(row,column)

      # Shape to train
      X_train = X_train.reshape((len(y_train),row,column))
      X_test = X_test.reshape((len(y_test),row,column))

      #Model
      #A lot of code but We build a model of 3 layers with 512 units.
      MAXLAYER = 3
      MAXUNITS = 9 # 2^MAXUNITS units per layer: 9 to have 512 units per layer.
      EPOCHS = 5000
      DROPOUT = 0.0
      OUTPUT = 10

      for LAYER in range(rLayer,(MAXLAYER+1),1):
        #print(LAYER)
        if(rFlag==0):
          rLayer=3
          rUnits=9 # 2^MAXUNITS units per layer
        else:
          rFlag=0
        for x in range(rUnits, (MAXUNITS+1),1): # ten*2 files per kind of layer
          #print(2**x)
          UNITS = (2**x)
          
          def create_model():
            def get_lr_metric(optimizer):
              def lr(y_true, y_pred):
                  return optimizer.learning_rate
              return lr
              # create static model
              # You will see the LSTM requires the input shape of the data it is being given.
            #simple_lstm_model = tf.keras.models.Sequential([
            #   tf.keras.layers.LSTM(units=UNITS,return_sequences=True,input_shape=X_train.shape[-2:],dropout=DROPOUT),
              #  tf.keras.layers.LSTM(units=UNITS),
              # tf.keras.layers.Dense(units=UNITS, activation='relu'),
                #tf.keras.layers.Dense(25,activation='softmax')])
              # create dynamic model
            model = None
            model = tf.keras.models.Sequential()

            if (LAYER == 1):
                model.add(tf.keras.layers.LSTM(units=UNITS,return_sequences=False,input_shape=X_train.shape[-2:],dropout=DROPOUT))
            else:
                for i in range(0,LAYER,1):
                    if (i == 0):
                        model.add(tf.keras.layers.LSTM(units=UNITS,return_sequences=True,input_shape=X_train.shape[-2:],dropout=DROPOUT))
                    else:
                        if ((i+1 == LAYER)):
                            model.add(tf.keras.layers.LSTM(units=UNITS,dropout=DROPOUT))
                        else:
                            model.add(tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=DROPOUT))
            model.add(tf.keras.layers.Dense(units=UNITS, activation='relu'))
            model.add(tf.keras.layers.Dense(OUTPUT,activation='softmax'))
                # Compile model
            opt = None
            opt = tf.keras.optimizers.Adam(1e-3)
            lr_metric = None
            lr_metric = get_lr_metric(opt)
            model.compile(optimizer=opt,
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                                      metrics=['acc', lr_metric])
            #model.summary()
            return model

          #h = tf.keras.callbacks.History()
          cbks = None  
          cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), #(1e-3)/(epoch+1) #0.001
                  #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                  tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                  tf.keras.callbacks.CSVLogger('/home/est1/Edgar/tensorflowPractice/tfLSTMSupervisedModelEvaluation/mnist/dynamic/log_'+formReadPixel+'_'+str(row)+'x'+str(column)+'_L'+str(LAYER)+'_U'+str(UNITS)+'.csv', append=True, separator=',')
                  ]#(write_graph=True)] 

          #Initializing a model of 3 layers with 512 units.
          modelAslan = None        
          modelAslan = KerasClassifier(build_fn=create_model, epochs=EPOCHS, batch_size=256)
                        
          ##GridSearchCV
          gs = None
          gs = GridSearchCV(estimator=modelAslan, 
                                param_grid=[{'epochs': [EPOCHS]}],
                                  refit=True, # refit makes to seem k-fold a k+1 fold because after the k-folds it allows refit the model with the whole trainingn dataset  to yeld a model with the best parameters
                                  scoring='accuracy',
                                  cv=5
                                )
          
          gs = gs.fit(X=X_train,
                      y=y_train,
                      callbacks=cbks, 
                      )
          #Save model
          # save the model to disk
          gs.best_estimator_.model.save_weights('/home/est1/Edgar/tensorflowPractice/tfLSTMSupervisedModelEvaluation/mnist/dynamic/log_'+formReadPixel+'_'+str(row)+'x'+str(column)+'_L'+str(LAYER)+'_U'+str(UNITS)+'.h5')
          #time.sleep(60)
          #gs.best_estimator_.model.load_weights('/content/drive/My Drive/PhD/Edgar/tensorflowPractice/tfLSTMSupervisedModelEvaluation/handMnist/dynamic/log_L'+str(LAYER)+'_U'+str(UNITS)+'.h5')
          #loaded_model = gs
                    
          #result = loaded_model.score(X_test, y_test)
          #print((result*100), '.2f')
          #Save model

          CVA = gs.best_score_
          TA = gs.score(X_test, y_test)
          CVA = format((CVA*100), '.2f')
          TA = format((TA*100), '.2f')

          #get epochs per Layer, units

          #UNITS = (2**x)
          
          path = '/home/est1/Edgar/tensorflowPractice/tfLSTMSupervisedModelEvaluation/mnist/dynamic/log_'+formReadPixel+'_'+str(row)+'x'+str(column)+'_L'+str(LAYER)+'_U'+str(UNITS)+'.csv'
              
          df = pd.read_csv(path)
          #df.shape
                  
          #Train Data version1
          epochs = np.array(df.values[:,0])
          
          z = np.where(epochs == 0)
          z = np.asarray(z)
          z = z.ravel()
          fold1 = epochs[z[0]:(z[1])]
          fold2 = epochs[z[1]:(z[2])]
          fold3 = epochs[z[2]:(z[3])]
          fold4 = epochs[z[3]:(z[4])]
          fold5 = epochs[z[4]:(z[5])]
          epochsWholeTrain = epochs[z[5]:]

          folds = np.array([(max(fold1)+1),(max(fold2)+1),(max(fold3)+1),(max(fold4)+1),(max(fold5)+1)])

          #print("cv mean epochs",l,UNITS, np.mean(folds))
          #print(np.mean(folds))
          #print(np.mean(folds), max(epochsWholeTrain+1))

          #print(LAYER, UNITS, np.mean(folds), CVA, max(epochsWholeTrain+1),TA)
          amodel = [LAYER, UNITS, np.mean(folds), CVA, max(epochsWholeTrain+1),TA]
          import csv
          csv.register_dialect("hashes", delimiter=",")
          f = open('/home/est1/Edgar/tensorflowPractice/tfLSTMSupervisedModelEvaluation/mnist/dynamic/log_'+formReadPixel+'_'+str(row)+'x'+str(column)+'.csv','a')

          with f:
              #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
              writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
              writer.writerow(amodel) 
    break

# Original function to go through an image mnist in a spral way.
def spiral(DataExamples):
    N = len(DataExamples[0,:])
    for j in range(len(DataExamples)):
    
        mAux = np.empty((1,0), float)  # It is generated a empty matrix with shape (a,b)
        for i in range(int(N/2)): # range(one of the side of a square matrix 28/2 mnist)
                
            mAux = np.concatenate((mAux, DataExamples[j,i,i:(-1-i)]), axis=None)
            mAux = np.concatenate((mAux, DataExamples[j,i:(-1-i),(-1-i)]), axis=None)    
            mAux = np.concatenate((mAux, np.flip(DataExamples[j,(-1-i),(1+i):(N-i)], axis=None)), axis=None)    
            mAux = np.concatenate((mAux, np.flip(DataExamples[j,(i+1):(N-i),i], axis=None)), axis=None)
            
        DataExamples[j] = mAux.reshape(N,N)
    return DataExamples

#####Uncomment the Horizontal, Vertical or Espiral sections as required.#####

#Horizontal
#X_train,X_test,y_train,y_test = readData()
#plt.imshow(X_train[10])
# pendiente 392 y 784
#rowColumnInputShape(X_train,X_test,y_train,y_test,rRow=784,rFlag=1,rLayer=3,rUnits=9,formReadPixel ='H')
#Horizontal

#Vertical
X_train,X_test,y_train,y_test = readData()
# Rotate the images 90 degrees cloclwise
for i in range(len(X_train)): 
    X_train[i] = np.rot90(X_train[i],k=3)

for i in range(len(X_test)): 
    X_test[i] = np.rot90(X_test[i],k=3)

#plt.imshow(X_train[10])
rowColumnInputShape(X_train,X_test,y_train,y_test,rRow=196,rFlag=1,rLayer=3,rUnits=9,formReadPixel ='V')
#Vertical

#Espiral
#X_train,X_test,y_train,y_test = readData()
# Images with a spiral form
#X_train = spiral(X_train)
#X_test = spiral(X_test)
#plt.imshow(X_train[10])
#rowColumnInputShape(X_train,X_test,y_train,y_test,rRow=784,rFlag=1,rLayer=3,rUnits=9,formReadPixel ='S')
#Espiral
