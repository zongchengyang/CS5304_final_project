
# coding: utf-8

# # CS5304 Final Project
# ## Part 2 - Modified Model
# ### teammember: Shang Zhou - sz536, Zongcheng Yang - zy338

# ## Initialization

# In[37]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2


# In[38]:


img = cv2.imread("image.jpg")
plt.imshow(img)
plt.show()


# In[39]:


lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv2.inRange(img, lower, upper)
shapeMask.shape


# In[40]:


# find the contours in the mask
cnts, _ = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I found %d black shapes" % (len(cnts)))
cv2.imshow("Mask", shapeMask)
 
# loop over the contours
for c in cnts:
# draw the contour and show it
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


# In[41]:


get_ipython().magic('matplotlib inline')
import os 
from scipy import ndimage
from subprocess import check_output
import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[42]:


rows, cols= 350, 425
im_array = cv2.imread('data/train/LAG/img_00091.jpg', 0)
template = np.zeros([rows, cols], dtype='uint8') # initialisation of the template
template[:, :] = im_array[100:450, 525:950] # I try multiple times to find the correct rectangle. 
#template /= 255.
plt.subplots(figsize=(10, 7))
plt.subplot(121), plt.imshow(template, cmap='gray') 
plt.subplot(122), plt.imshow(im_array, cmap='gray')


# In[124]:


file = 'data/train/LAG/img_01512.jpg' # img_00176,img_02758, img_01512
img = cv2.imread(file, 0) 
img2 = img
width, height = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv2.rectangle(img, top_left, bottom_right, 255, 3)
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplot(121),plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img, cmap ='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


# In[76]:


method = eval('cv2.TM_CCOEFF')
indexes=[1, 30, 40, 5]

train_path = "data/train/"
sub_folders = [d for d in os.listdir(train_path)]
for sub_folder in sub_folders:
    if sub_folder == ".DS_Store":
        continue
    files = [f for f in os.listdir(train_path + sub_folder)]
    k = 0
    _, ax = plt.subplots(2,2,figsize=(10, 7))
    for file in [files[x] for x in indexes]: # I take only 4 images of each group. 
        img = cv2.imread(train_path + sub_folder + "/" + file, 0)
        img2 = img
        width, height = template.shape[::-1]
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
 
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        if k == 0: 
            ax[0, 0].imshow(img,cmap = 'gray')
            plt.xticks([]), plt.yticks([])
        if k == 1: 
            ax[0, 1].imshow(img,cmap = 'gray')
            plt.xticks([]), plt.yticks([])
        if k == 2: 
            ax[1, 0].imshow(img,cmap = 'gray')
            plt.xticks([]), plt.yticks([])
        if k == 3: 
            ax[1, 1].imshow(img,cmap = 'gray')
            plt.xticks([]), plt.yticks([])
        k += 1
    plt.suptitle(sub_folder)
    plt.show()


# In[118]:


def Sample_Plot(annotation_path, label, image, top_left, bottom_right, i):
    img = cv2.imread(annotation_path + label + '/' + image , 0)
    cv2.rectangle(img, top_left, bottom_right, 255, 3)
    #fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplot(3, 2, i)
    plt.imshow(img, cmap ='gray')
    plt.title(label)
    plt.xticks([])
    plt.yticks([])


# In[119]:


import json
from pprint import pprint

annotation_path = 'data/preprocess_train/'
labels = ['DOL', 'BET', 'SHARK', 'YFT', 'LAG', 'ALB']
top_lefts = []
bottom_rights = []
images = []

for label in os.listdir(annotation_path):
    with open(annotation_path + label) as file:    
        data = json.load(file)
    top_lefts.append((int(data[1]["annotations"][0]['x']), int(data[1]["annotations"][0]['y'])))
    bottom_rights.append((int(data[1]["annotations"][1]['x']), int(data[1]["annotations"][1]['y'])))
    images.append(data[1]["filename"])

fig, ax = plt.subplots(3, 2, figsize=(12, 10))
for i, label in enumerate(labels):
    Sample_Plot(train_path, label, images[i], top_lefts[i], bottom_rights[i], i + 1)
plt.show()


# In[123]:


threshold1 = 2000
threshold2 = 5000


# In[129]:


mat = (img[:200, :125] - img[-200:, -125:]).reshape(25000)**2


# In[130]:


np.sqrt(sum(i for i in mat) / sum(1 for i in mat))

