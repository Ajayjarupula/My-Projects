#!/usr/bin/env python
# coding: utf-8

# # Develop a neural network that can read handwriting

# # Importing Libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[7]:


#A popular machine learning library, TensorFlow, to recognize handwritten digits from the MNIST dataset


# # Loading Dataset

# In[2]:


# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]


# In[3]:


# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# In[4]:


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[5]:


# Train the model
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5)


# In[6]:


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test, verbose=2)
print('\nTest accuracy:', test_acc)


# In[ ]:




