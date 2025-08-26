from ultralytics import YOLO
import cv2 as cv

model = YOLO("/home/abisheck/python/Object_detection_basics/yolo_weights/yolov8n.pt")

results = model("/home/abisheck/Downloads/vehicle.jpg")


for r in results:
    img = r.plot() 
    
    cv.imshow("YOLO Detection", img)  
    cv.waitKey(0) 
    cv.destroyAllWindows()