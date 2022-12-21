import cv2
import numpy as np
import time

# YOLOv3  is used to solve the object detection problem
# There are 4 versions of YOLOv3 model: tiny, 320, 416 and 608
# As the number grows, the accuracy increase but at the same time FPS decreases

startTime = time.time()
displayTime = 5       #Display time period
fps = 0
#cap = cv2.VideoCapture(0)  #Laptop webcam is used

#cap = cv2.VideoCapture("D:\PycharmProjects\pythonProject3\\traffic_sunny_1080.mp4") # High quality sunny video
#cap = cv2.VideoCapture("D:\PycharmProjects\pythonProject3\\traffic_sunny_480.mp4")  # Low quality sunny video
#cap = cv2.VideoCapture("D:\PycharmProjects\pythonProject3\\traffic_night_1080.mp4") # High quality night video
#cap = cv2.VideoCapture("D:\PycharmProjects\pythonProject3\\traffic_night_480.mp4")  # Low quality night video
#cap = cv2.VideoCapture("D:\PycharmProjects\pythonProject3\\traffic_rainy_1080.mp4") # High quality rainy video
#cap = cv2.VideoCapture("D:\PycharmProjects\pythonProject3\\traffic_rainy_480.mp4")  # Low quality rainy video


whT = 320 # w:width, h:height, T:target
confidenceThreshold = 0.35 # Confidence level threshold
nmsThreshold = 0.3  # The lower the nmsThreshold value, the lower the number of boxes for detected objects
# Load Model
classesFile = 'coco.names' # Open the file that contains the class names of the model
classNames = []

with open(classesFile, 'rt') as f:
    classNames =f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

# Create network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)    # Use cv2 for backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)   # Use CPU as target

def findObjects(outputs, img):
    hT, wT, cT = img.shape     # hT: height, wT: width, cT: channel
    bbox = []                  # bounding box
    classIds = []              # class Ids of objects
    confidences = []                 # confidence of an object exists

    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Take the first five element of the output since the rest is zeros
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w,h = int(detection[2]*wT), int(detection[3]*hT)  # Convert width and height values into pixels
                x,y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)  # Since x and y coordinates are the center, find corners
                bbox.append(([x,y,w,h]))
                classIds.append(classId)
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confidences, confidenceThreshold, nmsThreshold) #Indices of the detected objects
    #print(indices)

    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,128), 2)  # idx0: Used Image, idx1: x and y coords,
                                                               # idx2: corner points, idx3: RGB color, idx4: thickness
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidences[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        #print(f'{classNames[classIds[i]].upper()} {int(confidences[i]*100)}%')
    #print("********************")

while True:
    success, img = cap.read() # store the captured image to the img
    img = cv2.resize(img, None, fx=1.0, fy=1.0  , interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #outputNames = net.getUnconnectedOutLayers() #-> Gives the indexes of the output layers
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()] #yolo_16 and yolo_23 are the output layers

    outputs = net.forward(outputNames)
    #print(outputs[0].shape) # Shape of the first output: (300,85)
    #print(outputs[1].shape) # Shape of the second output: (1200,85)
    #print(outputs)

    findObjects(outputs, img)

    cv2.imshow("Image", img) # show the image to be displayed
    cv2.waitKey(1)          # wait for 1 ms

    fps += 1
    TIME = time.time() - startTime
    if TIME > displayTime:
        print(fps/TIME)
        fps = 0
        startTime = time.time()

