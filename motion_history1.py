import argparse
import sys
import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

MHI_DURATION = 10
DEFAULT_THRESHOLD = 32


def get_model(input_shape, output_shape):
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPool2D())

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPool2D())

  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPool2D())

  model.add(Flatten())

  model.add(Dense(64, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(output_shape, activation='softmax'))

  return model


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("model_dir")
  args = parser.parse_args()

  import sys
  try:
    video_src = sys.argv[1]
  except:
    video_src = './real/real_%d.jpg'

  cv2.namedWindow('motion-history')
  cv2.namedWindow('raw')
  cv2.moveWindow('raw', 200, 0)
  i = 0

  # Currently available possible actions. Order matters here because, training
  # procedure done with this order.
  ACTIONS_DICT = {0: "EL SALLIYOR", 1: "HAREKETLI", 2: "BOKS", 3: "DURGUN"}

  input_shape = (120, 120, 1)
  output_shape = len(ACTIONS_DICT)

  model = get_model(input_shape, output_shape)
  model = keras.models.load_model(args.model_dir)

  predictions = []

  while True:

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
      mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
      cv2.imshow('motempl', mh)
      cv2.imshow('raw', frame)

      prev_frame = frame.copy()
      p_img = cv2.resize(mh, (120, 120))

      prediction = model.predict(np.array([np.expand_dims(p_img, axis=3)]))
      p = np.argmax(prediction)

      predictions.append(p)
      # We collect 20 frame(prediction on frame) to understand the current move.
      # Then we count each prediction and take the argmax as final output.
      if len(predictions) == 20:

        pred = np.bincount(np.array(predictions))
        pred = np.argmax(pred)
        print(ACTIONS_DICT[pred])
        predictions = []

      if 0xFF & cv2.waitKey(5) == 97:
        break

  cv2.destroyAllWindows()

  pass


if __name__ == "__main__":
  main()
