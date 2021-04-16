import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def create_dataset_element(base_size,end_size,magnitude_min,magnituge_max):
  array = np.random.rand(base_size,base_size)
  coef = np.random.permutation(np.arange(magnitude_min,magnituge_max,0.1))[0]
  element = cv2.resize(array, dsize=(end_size,end_size), interpolation=cv2.INTER_CUBIC)
  element = element*coef
  if np.min(element)>=0:
      min_value = np.min(element)
      element = element - min_value
  else:
      min_value = np.min(element)
      element = element + abs(min_value)
  return element
  

def wraptopi(input):
  pi = 3.1415926535897932384626433;
  output = input - 2*pi*np.floor( (input+pi)/(2*pi) );
  return (output)
  
