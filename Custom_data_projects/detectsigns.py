import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")


cap = cv2.VideoCapture(0)


log_file = open("detections.txt", "a")  


cv2.namedWindow("Hand Sign Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Sign Detection", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    results = model(frame)
    
  
    annotated_frame = results[0].plot()
    
  
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  
        cls_name = model.names[cls_id]  
        print(f"Detected: {cls_name}")
        
      
        log_file.write(f"{cls_name}\n")
    
   
    cv2.imshow("Hand Sign Detection", annotated_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
