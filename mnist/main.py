
# coding: utf-8


from keras.utils import np_utils
import numpy as np
np.random.seed(10)

from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()

x_train = x_train_image.reshape(60000,784).astype('float64')
x_test = x_test_image.reshape(10000,784).astype('float64')

x_train_normalize = x_train/255
x_test_normalize = x_test/255

y_train_onehot =np_utils.to_categorical(y_train_label)
y_test_onehot = np_utils.to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


train_histoy =model.fit(x=x_train_normalize,
                         y=y_train_onehot,validation_split=0.2, 
                         epochs=10, batch_size=100,verbose=2)
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()

show_train_history(train_histoy,'acc','val_acc')

show_train_history(train_histoy,'loss','val_loss')

scores = model.evaluate(x_test_normalize, y_test_onehot)
print('accuracy=',scores[1])
prediction=model.predict_classes(x_test)


import matplotlib.pyplot as plt
plot_images_labels_prediction(x_test_image,y_test_label,
                              prediction,idx=340)
import pandas as pd
pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])
plot_images_labels_prediction(x_test_image,y_test_label,
                              prediction,idx=2035,num=1)

