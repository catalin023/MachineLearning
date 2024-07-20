#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


# In[2]:


from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        #citim imaginea
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        #returnam numele imaginii si imaginea
        yield filename, img


def read_data_set(images_path, csv_path):
    label_data = pd.read_csv(csv_path)
    #initializam un array de 0-uri cu dimensiunea imaginii 80x80x3
    images = np.zeros((len(label_data), 19200), dtype=np.float32)

    #incarcam imaginile din folder
    for filename, img in load_images_from_folder(images_path):
        #extragem numele imaginii
        file_name = os.path.splitext(filename)[0]
        #facem reshape la imagine in 1D si normalizam valorile pixelilor sa fie intre 0 si 1
        norm_img = (np.reshape(img, -1))/255.0
        
        #daca imaginea are doar un canal de culoare, o replicam de 3 ori ca sa avem o dimensiune constanta pe date
        if len(norm_img) == 6400:
            norm_img = np.tile(processed_img, 3)

        #cautam indexul imaginii in dataframe si o adaugam in arrayul de imagini
        index = label_data[label_data["image_id"] == file_name].index.tolist()
        if index:
            images[index[0]] = norm_img

    return images, label_data


# In[3]:


#citim datele de antrenare, validare si test
train_images, train_data = read_data_set("/kaggle/input/ml-competition2024/train", "/kaggle/input/ml-competition2024/train.csv")
valid_images, valid_data = read_data_set("/kaggle/input/ml-competition2024/validation", "/kaggle/input/ml-competition2024/validation.csv")
test_images, test_data = read_data_set("/kaggle/input/ml-competition2024/test", "/kaggle/input/ml-competition2024/test.csv")


# In[4]:


#redmiensionam imaginile la dimensiunea originala 80x80x3 pentru nn
train2_images = train_images.reshape((10500, 80, 80, 3))
valid2_images = valid_images.reshape((3000, 80, 80, 3))
test2_images = test_images.reshape((4500, 80, 80, 3))


# In[22]:


#plotarea imaginilor cu un anumit label
for i, label in enumerate(train_data["label"]):
    if label == 1:
        plt.figure(figsize=(5, 5))
        plt.title(f"Label: {label}")
        plt.imshow(train2_images[i])
        plt.axis('off')
        plt.show()


# KNN

# In[5]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# In[9]:


#deffinim un model knn simplu cu 5 vecini
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=5)
knn_model.fit(train_images, train_data["label"])


# In[10]:


#predictia pe datele de validare
pred = knn_model.predict(valid_images)


# In[11]:


#raportul de clasificare
print(classification_report(valid_data["label"], pred))


# In[7]:


#definim o functie pentru antrenare mai multor modele knn cu diferiti hyperparametrii
def knn(n_neighbors, alg):
    knn_model = KNeighborsClassifier(n_neighbors = n_neighbors, algorithm = alg, n_jobs=5)
    knn_model.fit(train_images, train_data["label"])
    
    prediction = knn_model.predict(valid_images)
    return knn_model, prediction


# In[8]:


#antrenam mai multe modele knn cu diferiti hyperparametrii
for n_neighbors in [13, 21, 51]:
    for alg in ["auto", "ball_tree", "kd_tree", "brute"]:
        knn_model, prediction = knn(n_neighbors, alg)
        print("nr: ", n_neighbors, " alg: ", alg)
        pred = knn_model.predict(valid_images)
        print(classification_report(valid_data["label"], pred))
        


# SVM

# In[13]:


from sklearn.svm import SVC


# In[14]:


#definim un model svm simplu
svm_model = SVC(verbose = 1, kernel="rbf", C=1)
svm_model.fit(train_images, train_data["label"])


# In[16]:


#facem predictia pe datele de validare si afisam raportul de clasificare
prediction = svm_model.predict(valid_images)
print(classification_report(valid_data["label"], prediction))


# write submission.csv

# In[ ]:


#scriem predictiile pe datele de test intr-un fisier csv
submission_df = pd.DataFrame({"image_id": test_data["image_id"], "label": prediction})
submission_df.to_csv("submission_svc.csv", index=False)


# In[ ]:


#definim o functie pentru antrenarea mai multor modele svm cu diferiti hyperparametrii
def svm(kernel, c):
    svm_model = SVC(kernel=kernel, C=c)
    svm_model.fit(train_images, train_data["label"])
    
    prediction = svm_model.predict(valid_images)
    return svm_model, prediction


# In[ ]:


#antrenam mai multe modele svm cu diferiti hyperparametrii
for kernel in ['linear', 'poly', 'rbf']:
    for c in [1, 10, 100]:
        svm_model, prediction = svm(kernel, c)
        print("kernel: ", kernel, " c: ", c)
        pred = svm_model.predict(valid_images)
        print(classification_report(valid_data["label"], pred))


# Neural net
# 

# In[6]:


import tensorflow as tf


# In[8]:


#crearea unui model simplu de nn cu 3 layere dense folosind libraria keras
nn_model = tf.keras.Sequential((
      tf.keras.layers.Dense(32, activation='relu', input_shape = (19200,)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      #layer cu 3 neuroni pentru cele 3 clase
      tf.keras.layers.Dense(3, activation='softmax')
  ))
nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#antrenarea modelului pe datele de training
history = nn_model.fit(
  train_images, train_data["label"], epochs = 100, batch_size=32, validation_split=0.2
)


# In[9]:


#facem predictia pe datele de validare
prediction = nn_model.predict(valid_images)
prediction_max = []
for pred in prediction:
    #extragem indexul clasei cu probabilitatea maxima
    prediction_max.append(np.argmax(pred))
print(classification_report(valid_data["label"], prediction_max))


# In[8]:


def plot_history(history, file = "plot.png"):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(file) 
    
    plt.show()


# In[ ]:


submission_df = pd.DataFrame({"image_id": test_data["image_id"], "label": prediction_max})
submission_df.to_csv("nn3_submission.csv", index=False)


# In[14]:


#definim un model mai complex de cnn cu 3 layere convolutionale si 3 layere dense
nn_model = tf.keras.Sequential((
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)),
    #layer de normalizare a datelor de iesire
    tf.keras.layers.BatchNormalization(),
    #layer de pooling pentru reducerea dimensiunii
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    #layer de flatten pentru a transforma datele intr-un vector 1D
    tf.keras.layers.Flatten(),
    
    #3 layere dense
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    #folosim functia de activare softmax pentru a obtine probabilitatile pentru cele 3 clase
    tf.keras.layers.Dense(3, activation='softmax')
))
nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.00005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(
  train2_images, train_data["label"], epochs = 100, batch_size=32, validation_split=0.2
)


# In[11]:


prediction = nn_model.predict(valid2_images)
prediction_max = []
for pred in prediction:
    prediction_max.append(np.argmax(pred))
print(classification_report(valid_data["label"], prediction_max))


# In[12]:


plot_history(history, "cnn.png")


# In[ ]:


import tensorflow as tf
import os

# Create the directory if it does not exist
if not os.path.exists('models'):
    os.makedirs('models')

nn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)),
    #layer de normalizare a datelor de iesire
    tf.keras.layers.BatchNormalization(),
    #layer de pooling pentru reducerea dimensiunii
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    #layer de flatten pentru a transforma datele intr-un vector 1D
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(3, activation='softmax')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#salvarea modelului in epoca cu cea mai buna acuratete pe datele de validare
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/best_model.keras', 
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

#antrenarea modelului cu callbackurile definite
nn_model.fit(
    train2_images, 
    train_data["label"], 
    epochs=400, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[checkpoint_callback]
)


# In[16]:


import tensorflow as tf
import numpy as np
import os

if not os.path.exists('models'):
    os.makedirs('models')


#definirea unui model mai complex de cnn cu 6 layere convolutionale si 3 layere dense
nn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(80, 80, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(3, activation='softmax')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#salvarea modelului in epoca cu cea mai buna acuratete pe datele de validare
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = nn_model.fit(
    train2_images, 
    train_data["label"], 
    epochs=400, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[checkpoint_callback]
)


# In[13]:


import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import os

if not os.path.exists('models'):
    os.makedirs('models')

#definirea unui model mai complex de cnn cu 8 layere convolutionale si 4 layere dense
nn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(80, 80, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
#     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(3, activation='softmax')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#salvarea modelului in epoca cu cea mai buna acuratete pe datele de validare
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = nn_model.fit(
    train2_images, 
    train_data["label"], 
    epochs=400, 
    batch_size=128, 
    validation_split=0.2, 
    callbacks=[checkpoint_callback]
)



# In[9]:


plot_history(history, "cnn3.png")


# In[10]:


#incarcam modelul antrenat
nn_model = tf.keras.models.load_model('/kaggle/working/models/best_model.keras')


# In[12]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

#generam matricea de confuzie
cm = confusion_matrix(valid_data["label"], prediction_max)

#plotarea matricei de confuzie
mat = ConfusionMatrixDisplay(confusion_matrix=cm)
mat.plot()

plt.savefig('confusion_matrix_cnn3_plot.png')

plt.show()

