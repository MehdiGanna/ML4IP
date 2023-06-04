# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:30:21 2023

@author: mehdi ganna
"""
import tensorflow as tf
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.optimizers import Adamax
from sklearn.preprocessing import LabelEncoder


#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#scanners
scanner_type = 'S'
resolution   = '500'

batch_size    = 32
img_height    = 128
img_width     = 128
channels      = 1
no_epochs     = 15
verbosity     = 1

save_model    = False
callback      = False


class MyCNN(Model):
    def __init__ (self, input_shape, nb_classes):
        num_filters = 32
        filter_size = 3
        pool_size = 2
        super().__init__()
        self.conv2d_finger = tf.keras.layers.Conv2D(num_filters, filter_size, input_shape=input_shape, padding='same')
        self.maxpooling2d_finger = tf.keras.layers.MaxPooling2D(pool_size=pool_size, trainable=True)
        self.flatten_finger = tf.keras.layers.Flatten()
        self.output_finger = tf.keras.layers.Dense(nb_classes[0], activation='softmax')
        
        self.conv2d_gender = tf.keras.layers.Conv2D(num_filters, filter_size, input_shape=input_shape, padding='same')
        self.maxpooling2d_gender = tf.keras.layers.MaxPooling2D(pool_size=pool_size, trainable=True)
        self.flatten_gender = tf.keras.layers.Flatten()
        self.output_gender = tf.keras.layers.Dense(nb_classes[1], activation='softmax')
        
        self.conv2d_race = tf.keras.layers.Conv2D(num_filters, filter_size, input_shape=input_shape, padding='same')
        self.maxpooling2d_race = tf.keras.layers.MaxPooling2D(pool_size=pool_size, trainable=True)
        self.flatten_race = tf.keras.layers.Flatten()
        self.output_race = tf.keras.layers.Dense(nb_classes[2], activation='softmax')
        
        self.conv2d_worktype = tf.keras.layers.Conv2D(num_filters, filter_size, input_shape=input_shape, padding='same')
        self.maxpooling2d_worktype = tf.keras.layers.MaxPooling2D(pool_size=pool_size, trainable=True)
        self.flatten_worktype = tf.keras.layers.Flatten()
        self.output_worktype = tf.keras.layers.Dense(nb_classes[3], activation='softmax')
        
        
    def call(self, inputs):
        #fingers
        conv2d_finger = self.conv2d_finger(inputs)
        maxpooling2d_finger = self.maxpooling2d_finger(conv2d_finger)
        flatten_finger = self.flatten_finger(maxpooling2d_finger)
        output_finger = self.output_finger(flatten_finger)
        
        #gender
        conv2d_gender = self.conv2d_gender(inputs)
        maxpooling2d_gender = self.maxpooling2d_gender(conv2d_gender)
        flatten_gender = self.flatten_gender(maxpooling2d_gender)
        output_gender = self.output_gender(flatten_gender)
        
        #race
        conv2d_race = self.conv2d_race(inputs)
        maxpooling2d_race = self.maxpooling2d_race(conv2d_race)
        flatten_race = self.flatten_race(maxpooling2d_race)
        output_race = self.output_race(flatten_race)
        
        #worktype
        conv2d_worktype = self.conv2d_worktype(inputs)
        maxpooling2d_worktype = self.maxpooling2d_worktype(conv2d_worktype)
        flatten_worktype = self.flatten_worktype(maxpooling2d_worktype)
        output_worktype = self.output_worktype(flatten_worktype)
        
        return [output_finger, output_gender, output_race, output_worktype]
        
             
def upload_image(PATH, FRGP, GENDER, RACE, WORK_TYPE):
    X   = []
    Y_1 = []
    Y_2 = []
    Y_3 = []
    Y_4 = []

    for i,j,k,l,m in zip(PATH, FRGP, GENDER, RACE, WORK_TYPE):        
        image = tf.keras.utils.load_img(i, target_size=(img_height,img_width), color_mode = "grayscale")
        input_arr = tf.keras.utils.img_to_array(image)/255.0
        X.append(input_arr)
        Y_1.append(j)
        Y_2.append(k)
        Y_3.append(l)
        Y_4.append(m)
        
    return np.array(X), [np.array(Y_1), np.array(Y_2), np.array(Y_3), np.array(Y_4)]

def plot_bar(data, title):
    counts = data.value_counts().rename_axis('categories').reset_index(name='frequencies')
    ax = sns.barplot(x='categories', y='frequencies', data=counts)
    ax.bar_label(ax.containers[0])
    plt.title(title)
    plt.show()

#reading pkl file (pandas from dataloader.py)
df = pd.read_pickle('./dataframes/df_'+scanner_type+'_'+resolution+'.pkl')
df = df.drop(df[(df.RACE == 'other') | (df.RACE == 'no answer') | 
                (df.WORK_TYPE == 'other') | (df.WORK_TYPE == 'no answer') ].index)
X   = df['PATH']
Y_1 = df['FRGP']
Y_2 = df['GENDER']
Y_3 = df['RACE'] 
Y_4 = df['WORK_TYPE']

#número de clases
num_classes   = [Y_1.nunique(), Y_2.nunique(), Y_3.nunique(), Y_4.nunique()]

#plotting output data
plot_bar(Y_1, "H I S T O G R A M A - F R G P")
plot_bar(Y_2, "H I S T O G R A M A - G E N D E R")
plot_bar(Y_3, "H I S T O G R A M A - R A C E")
plot_bar(Y_4, "H I S T O G R A M A - W O R K  T Y P E")


#uploading images
inputs, targets = upload_image(X.to_numpy(), Y_1.to_numpy(), 
                               Y_2.to_numpy(),Y_3.to_numpy(), Y_4.to_numpy())
#reshaping data
inputs          = inputs.reshape(inputs.shape[0], img_height, img_width, channels)

#one hot encoding
encoder    = LabelEncoder()
targets[1] = encoder.fit_transform(targets[1])
targets[2] = encoder.fit_transform(targets[2])
targets[3] = encoder.fit_transform(targets[3])

#spliting data: train + test
idx = int(inputs.shape[0]*0.8)
X_train, X_test = inputs[:idx], inputs[idx:]
y_train_1, y_test_1 = targets[0][:idx], targets[0][idx:]
y_train_2, y_test_2 = targets[1][:idx], targets[1][idx:]
y_train_3, y_test_3 = targets[2][:idx], targets[2][idx:]
y_train_4, y_test_4 = targets[3][:idx], targets[3][idx:]

#to categorical
y_train_1 = tf.keras.utils.to_categorical(y_train_1)
y_train_2 = tf.keras.utils.to_categorical(y_train_2)
y_train_3 = tf.keras.utils.to_categorical(y_train_3)
y_train_4 = tf.keras.utils.to_categorical(y_train_4)

y_test_1  = tf.keras.utils.to_categorical(y_test_1)
y_test_2  = tf.keras.utils.to_categorical(y_test_2)
y_test_3  = tf.keras.utils.to_categorical(y_test_3)
y_test_4  = tf.keras.utils.to_categorical(y_test_4)


y_train   = [y_train_1, y_train_2, y_train_3, y_train_4]
y_test    = [y_test_1, y_test_2, y_test_3, y_test_4]


if callback:
    # TensorBoard Visuals
    log_dir="./logs/log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#training model
model = None
model = MyCNN((img_height,img_width,channels), num_classes)
model.build((None,img_height,img_width,channels))
model.summary()

model.compile(optimizer=Adamax(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


if callback:
    # Fit data to model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epochs, 
                    validation_data=(X_test, y_test), verbose=verbosity, callbacks=[tensorboard_callback])
else:
    # Fit data to model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epochs, 
                        validation_data=(X_test, y_test), verbose=verbosity)

#save model
if save_model:
    model.save('./model/model_'+scanner_type+'_'+resolution+'.tf')
