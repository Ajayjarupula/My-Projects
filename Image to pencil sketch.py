#!/usr/bin/env python
# coding: utf-8

# # Image to Pencil Sketch with Python

# # Import Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[8]:


img = cv2.imread("C:/Users/admin/Downloads/wp1818433-ultron-wallpapers.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(img)
plt.show()


# # Converting an image into gray_scale image

# In[9]:


gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.axis('off')
plt.imshow(gray_img)
plt.show()


# # Inverting the Gray Image

# In[19]:


inv_gray_img = cv2.bitwise_not(gray_img)
plt.axis('off')
plt.imshow(inv_gray_img)
plt.show()


# # Blurring the inverted gray image

# In[11]:


blur_img = cv2.GaussianBlur(inv_gray_img,(21,21),0)
plt.axis('off')
plt.imshow(blur_img)
plt.show()


# # Inverting the blurred image

# In[12]:


inv_blur_img = cv2.bitwise_not(blur_img)
plt.axis('off')
plt.imshow(inv_blur_img)
plt.show()


# # Creating Pencil Sketch image

# In[13]:


pencil_img = cv2.divide(gray_img,inv_blur_img,scale = 256.0)
plt.axis('off')
plt.imshow(pencil_img)
plt.show()

