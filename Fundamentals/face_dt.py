import cv2 as cv
import numpy as np

img=cv.imread('/home/abisheck/Downloads/abi.jpeg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
haar_cascade=cv.CascadeClassifier('haar_face.xml')


face_rect = haar_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

# Merge overlapping detections
face_rect, weights = cv.groupRectangles(list(face_rect), groupThreshold=1, eps=0.2)

for (x, y, w, h) in face_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv.imshow('Detected Faces', img)
print("The number of faces found:", len(face_rect))
cv.waitKey(0)
