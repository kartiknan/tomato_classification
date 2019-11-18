
# coding: utf-8

# # Inference for the tomato model

# In[3]:


import numpy as np
import os
from fastai.vision import *


# ### Step 1: Setup the paths

# In[8]:


# This is the project path - modify as required
path = Path('C:/Users/kar16/Desktop/Fellowship_AI/tomato_classification')    # CHANGE HERE

# Under the path above there should be a folder called datasets - this is where all images are
# Do not change this
path_data = path/'datasets'

# Models are in the models folder - do not change this either
path_model = path/'models'


# ### Step 2: Load the model

# In[14]:


learn = load_learner(path_model, 'model.pkl')


# In[17]:


# The model can be viewed as follows. This is a resnet18 model, easy to find online
# Here is one: https://pytorch.org/hub/pytorch_vision_resnet/
# best to google resnet18, and look at the images
# print(learn.model)


# ### Step 3: Predict for a test image

# In[18]:


# The test image(s) should be stored in the datasets folder
# path_data.ls()


# In[28]:


# Read in image and convert it using open_image()
fn = 'not_tomato_6.jpg'
img = open_image(path_data/fn)


# In[29]:


prediction = learn.predict(img)[0]


# In[30]:


print('The predicted output is: ',prediction)
