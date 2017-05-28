# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:46:16 2017

@author: snate
"""

#Da fucking deepdream
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imsave
from IPython.display import Image, display
import inception5h

inception5h.maybe_download()

model = inception5h.Inception5h()

def convert_image(img):
    return np.float32(PIL.Image.open(img))

def save_image(img,name):
    img = (np.clip(img,0.0,255.0)).astype(np.uint8)
    with open(name,"wb") as file:
        PIL.Image.fromarray(img).save(file,'jpg')
        
def plot_image(img):
    img=np.clip(img,0.0,255.0)
    display(PIL.Image.fromarray(img.astype(np.uint8)))
    
def plot_gradients(grad):
    normed=normalize(grad)
    plt.imshow(normed,interpolation="bilinear")
    plt.show()
    
def normalize(input):
    min = input.min()
    max = input.max()
    return  (input-min)/(max-min)

def resize(img,size=None,factor=None):
    if factor is not None:
        size = (np.array(img.shape[0:2]) * factor).astype(int)
    else:
        size = size[0:2]
    size = tuple(reversed(size))
    img = (np.clip(img,0.0,255.0)).astype(np.uint8)
    img = PIL.Image.fromarray(img)
    resized = img.resize(size,PIL.Image.LANCZOS)
    return resized

def make_tiles(size,tile_size=400):
    num_tiles = int(round(size/tile_size))
    num_tiles = max(1,num_tiles)
    tile_size = math.ceil(size/num_tiles)
    return tile_size

def tiles_gradient(gradient,img,tile_size=400,tile_to_min_tile_ratio=4,normalize = True):
    grad = np.zeros_like(img)
    x,y,z = grad.shape
    x_tile = make_tiles(x,tile_size)//tile_to_min_tile_ratio
    y_tile = make_tiles(y,tile_size)//tile_to_min_tile_ratio
                       
    x_start_position=random.randint(-(tile_to_min_tile_ratio-1)*x_tile,-x_tile)
    
    img = np.asanyarray(img)
    
    while x_start_position < x:
        x_end_position = x_start_position+x_tile
        x_real_start = max(0,x_start_position)
        x_real_end = max(x_end_position,x_tile)
        
        y_start_position=random.randint(-(tile_to_min_tile_ratio-1)*y_tile,-y_tile)
        
        while y_start_position < y:
            y_end_position = y_start_position+y_tile
            y_real_start = max(0,y_start_position)
            y_real_end = max(y_end_position,y_tile)
            
            tile = img[x_real_start:x_real_end,y_real_start:y_real_end]
            
            feed_dict = model.create_feed_dict(image = tile)
            
            new_grad = session.run(gradient, feed_dict = feed_dict)
            
            if normalize:
                new_grad /= (np.std(new_grad) + 1e-8)
                
            grad[x_real_start:x_real_end,y_real_start:y_real_end,:] = new_grad
                
            y_start_position = y_end_position
            
        x_start_position = x_end_position
    
    return grad


def dream(layer,image,iterations=10,step_size=3.0,tile_size=400,show_gradient = False,smooth=True):
    
    img = image.copy()
    plot_image(img)
    
    gradient = model.get_gradient(layer)
    
    for i in range(iterations):
        
        grad = tiles_gradient(gradient,img)
        
        if smooth:
            sigma = (i*4.0)/iterations+0.5
            smooth1 = gaussian_filter(grad,sigma)
            smooth2 = gaussian_filter(grad,sigma*0.5)
            smooth3 = gaussian_filter(grad,sigma*2)
            
            grad = smooth1+smooth2+smooth3
            
            step_size_scaled = step_size/(np.std(grad)+1e-8)
            
            img+=step_size_scaled*grad
        
        else:
            img+=grad
            
        
        if show_gradient:
            plot_gradients(grad)
            
    
    plot_image(img)
    imsave("aaaa.jpg",img)
    return img
    
    
def recursive_dream(layer,image,repeats=4, rescale=0.7, blend = 0.2, iterations=10,step_size=3.0,tile_size=400,show_gradient = False,smooth=True,blur=True):
    if repeats > 0:
        sigma=0.5
        img = gaussian_filter(image,sigma=(sigma,sigma,0.0))
        img = resize(img,factor = rescale)
        
        image=np.asanyarray(image)
        
        result = recursive_dream(layer,img,repeats-1,rescale,blend, iterations,step_size,tile_size,show_gradient,smooth)
        upscaled =  resize(img=result,size=image.shape)
        upscaled = np.asanyarray(upscaled)
        image = blend*image+(1-blend)*upscaled
                         
    result = dream(layer,image,iterations,step_size,tile_size,show_gradient,smooth)
    
    return result
        
        
        
        
session = tf.InteractiveSession(graph=model.graph)


index = 3  #choose at which layer to dream
layer = model.layer_tensors[index]
#layer


image_path = "k.jpg" #choose what image to dream
image=convert_image(image_path)

#dream(layer,image,step_size=10)

recursive_dream(layer,image,show_gradient=True)
            
