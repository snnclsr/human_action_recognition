#!/usr/bin/env python
"""
Example to extract motion history images using opencv2. stripped from opencv2 python examples motempl.py
link to the gif: https://giphy.com/gifs/bJDYIRToRpkEU
command to extract the jpgs: convert example.gif -coalesce images/example-%03d.jpg
You have to use fixed length pattern for image sequence, such as ./images/example-%03d.jpg
"""

import sys
import argparse

import numpy as np
import cv2
import re

MHI_DURATION = 30
DEFAULT_THRESHOLD = 32


def toMH(filename, lines, personnum, filenum):

  # This should be fixed. :)
  video_src = './handclapping_frames/' + lines[((personnum - 1) * 4) + filenum - 1][0] + '_uncomp/frame%d.jpg'
  print(video_src)

  i = 0
  while i < 1:
    i += 1
    cam = cv2.VideoCapture(video_src)
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    j = 0
    while True:
      j += 1
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
      try:

        if j == int(lines[((personnum - 1) * 4) + filenum - 1][3]):

          cv2.imwrite(f'{filename}/' + str(personnum) + '_d' + str(filenum) + "_" + str(j) + '.png', mh)
        if j == int(lines[((personnum - 1) * 4) + filenum - 1][5]):
          cv2.imwrite(f'{filename}/' + str(personnum) + '_d' + str(filenum) + "_" + str(j) + '.png', mh)
        if j == int(lines[((personnum - 1) * 4) + filenum - 1][7]):
          cv2.imwrite(f'{filename}/' + str(personnum) + '_d' + str(filenum) + "_" + str(j) + '.png', mh)
        if j == int(lines[((personnum - 1) * 4) + filenum - 1][9]) - 1:
          cv2.imwrite(f'{filename}/' + str(personnum) + '_d' + str(filenum) + "_" + str(j) + '.png', mh)

      except:
        print("ATLANDI : " + 'box/' + str(personnum) + '_d' + str(filenum) + "_" + str(j) + '.png')
        continue
      prev_frame = frame.copy()
      if 0xFF & cv2.waitKey(5) == 27:
        break

  # cv2.destroyAllWindows()

  pass


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("filename")
  args = parser.parse_args()

  sequences_file = "sequences.txt"
  fh = open(sequences_file)

  lines = []
  for line in fh:
    line = line.strip('\n')
    line = re.split(' |,|\t|-', line)
    line = list(filter(None, line))
    if any(args.filename in s for s in line):
      lines.append(line)

  lines = list(filter(None, lines))
  # print(lines)

  for personnum in range(1, 26):
    for filenum in range(1, 5):
      toMH(args.filename, lines, personnum, filenum)


if __name__ == "__main__":
  main()
