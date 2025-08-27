from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np
from sort import*

#tracker
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
cap=cv.VideoCapture('/home/abisheck/Downloads/cars.mp4')
# cap.set(3,1280)
# cap.set(4,720)
Limits = [500,397,1593,397]
total_counts=[]
model = YOLO("/home/abisheck/python/Object_detection_basics/yolo_weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv.imread("/home/abisheck/Downloads/mask.png")
while True:
    success,img = cap.read()
    imgregion=cv.bitwise_and(img,mask)
    results=model(imgregion,stream=True)

    detection = np.empty((0,5))

    for r in results:
        boxes =r.boxes
        for box in boxes:
           x1,y1,x2,y2= box.xyxy[0]
           x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
           print(x1,y1,x2,y2)
           cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
           conf = math.ceil((box.conf[0]*100))/100
           
           cls=int(box.cls[0])
        #    cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),(max(35,y1))),offset=3)
           currentArray = np.array([x1,y1,x2,y2,conf])
           detection = np.vstack((detection,currentArray))
    resulttracker=tracker.update(detection)
    # cv.line(img,(Limits[0],Limits[1]),(Limits[2],Limits[3]),(255,0,0),thickness=8)
    for result in resulttracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h = x2-x1,y2-y1
        # cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),(max(35,y1))),offset=10,thickness=3,scale=2)
        cx,cy= x1+w/2,y1+h/2
        cv.circle(img,(int(cx),int(cy)),5,(255,0,255),cv.FILLED)

        if Limits[0]<cx<Limits[2] and Limits[1]-30<cy<Limits[1]+30:
            if total_counts.count(id) == 0:
                total_counts.append(id)
    cvzone.putTextRect(img,f' count is: {len(total_counts)}',(50,50),offset=10,thickness=3,scale=2)
    cv.imshow('image',img)
    
    if cv.waitKey(0) & 0xFF==ord('d'):
        break


