#!/usr/bin/env python


"""
Example to extract motion history images using opencv2. stripped from opencv2 python examples motempl.py
link to the gif: https://giphy.com/gifs/bJDYIRToRpkEU
command to extract the jpgs: convert example.gif -coalesce images/example-%03d.jpg
You have to use fixed length pattern for image sequence, such as ./images/example-%03d.jpg
"""

import sys
import os.path as osp
from firebase import firebase
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, ReLU, MaxPool2D




MHI_DURATION = 10
DEFAULT_THRESHOLD = 32
def main():
  import sys
  try: video_src = sys.argv[1]
  except: video_src = './real/real_%d.jpg'

  cv2.namedWindow('motion-history')
  cv2.namedWindow('raw')
  cv2.moveWindow('raw', 200, 0)
  i = 0

  #model = get_model_HW_R_B_HC()
  #model = keras.models.load_model("model_HW_R_B_HC.h5")
  last_send = ""
  c = 700

  while True :


    cam = cv2.VideoCapture(0)

    ret, frame = cam.read()

    h, w = frame.shape[:2]

    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0

    while True:
      ret, frame = cam.read()
      if not ret:
        break
      frame_diff = cv2.absdiff(frame, prev_frame)
      gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
      ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
      timestamp += 1

      # update motion history
      cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

      # normalize motion history
      mh = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
      cv2.imshow('motempl', mh)
      cv2.imshow('raw', frame)


      prev_frame = frame.copy()

      #print(mh.shape)
      """
      p_img = cv2.resize(mh, (120, 120))
      #print("p_img shape: ", p_img.shape)
      prediction = model.predict(np.array([np.expand_dims(p_img, axis=3)]))
      #print(prediction)
      #print(np.argmax(prediction))

      pred = np.argmax(prediction)
      """
      """
      if not prediction[prediction > 0.85]:
        print("DURGUN: ", prediction)
      elif pred == 1:
        print("HAREKETLI: ", prediction[0][0])
        if last_send != pred:
          #firebase.put('foto','hareket',"kosuyor")
          last_send = pred
      elif pred == 0:
        print("EL SALLIYOR: ", prediction[0][1])
        if last_send != pred:
          firebase.put('foto','hareket',"el_Sallıyor")
          last_send = pred
      elif pred == 2:
        print("BOKS:", prediction[0][2])
        if last_send != pred:
          firebase.put('foto','hareket',"boks")
          last_send = pred
      elif pred == 3:
        print("ALKIS: ", prediction[0][3])
      """

      
      
      if 0xFF & cv2.waitKey(5) == 97:
        cv2.imwrite("durgun_veri/" + str(c) + ".png", cv2.resize(mh, (120, 120)))
        c += 1        

  cv2.destroyAllWindows()

  pass

if __name__ == "__main__":
  main()

#vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
