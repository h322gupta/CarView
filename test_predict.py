# -*- coding: utf-8 -*-
"""
file : test_predict.py

usage : python3 test_predict.py --testDir testdir
output : None
saved files : dataframe hainvg the name of the image and predicted class along with probability

"""

import pandas as pd
import os 
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

class Prediction:
  def __init__(self, testDir , modelPath = 'model_2.tflite'):
    self.dataDir = testDir
    self.imgs = os.listdir(testDir)
    self.model = tf.lite.Interpreter(model_path = modelPath)

    self.labels = ['Rear', 'Front Left', 'Rear Right', 'Front', 'Front Right', 'Rear Left', 'Other']

  def imgPrep(self):
    img_height = 224
    img_width = 224
    img_array = {}
    for filename in self.imgs:
      img_path = os.path.join(self.dataDir, filename)
      img = keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
      img_array[filename] = keras.preprocessing.image.img_to_array(img) / 255.0

    return img_array


  def getPrediction(self,img_array):

    interpreter = self.model
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = [img_array]
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = self.labels[np.argmax(output_data)]
    return predicted_class , max(output_data[0])


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--testDir',default='testData',type=str)
  parser.add_argument('--modelDir',default='',type=str)
  args = parser.parse_args()


  obj = Prediction(args.testDir)
  # dataDir = '/content/drive/MyDrive/cars/training_data_v2/rightRear'
  # modelDir = '/content/drive/MyDrive/cars/model.tflite'
  # obj = Prediction(dataDir,modelDir)

  img_array = obj.imgPrep()

  imgs = img_array.keys()
  predClass = []
  imgNames = []
  prob = []
  for key , val in tqdm(img_array.items()):
    imgNames.append(key)
    pred , prob1 = obj.getPrediction(val)
    predClass.append(pred)
    prob.append(prob1)


  df = pd.DataFrame()
  df['imgName'] = imgNames
  df['predClass'] = predClass
  df['prob'] = prob

  df.to_csv("carView.csv")

