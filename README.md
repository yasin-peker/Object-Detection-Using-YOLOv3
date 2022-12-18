# Object-Detection-Using-YOLOv3

In this project, 4 different YOLOv3 models with different accuracy in different FPS values were created in the image processing area. The YOLOv3 models used are YOLOv3-tiny, YOLOv3-320, YOLOv3-416 and YOLOv3-618, respectively. These models have the highest accuracy rates on the YOLOv3-618 and the lowest accuracy on the YOLOv3-tiny. However, YOLOv3-tiny has the highest FPS values, while YOLOv3-618 has the lowest FPS values. For this reason, the YOLOv3-618 model should be used in applications where high performance and high accuracy are required, and the YOLOv3-tiny model should be used for minimum system requirements and maximum FPS values. Especially on mobile platforms such as Raspberry Pi and Arduino, the YOLOv3-tiny model performs much better.

![comparisons](https://user-images.githubusercontent.com/68354896/208297972-50f07aaf-c433-4eb9-8021-d201d76eae38.png)

![performances](https://user-images.githubusercontent.com/68354896/208297975-3642c411-8ae2-416f-9759-beb66221c440.png)

All of these models can find 80 different objects with a high accuracy rate. Table of objects that the model can detect:
![names-table](https://user-images.githubusercontent.com/68354896/208297962-e3c6c332-5485-41b2-a824-9f9faafd514c.png)

Weights of the YOLOv3 models can be downloaded from https://pjreddie.com/darknet/yolo/
