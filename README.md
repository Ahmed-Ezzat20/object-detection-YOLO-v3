# object-detection-YOLO-v3
In this project we are going to learn how to run one of the most popular object detection algorithms YOLOv3.

## What is YOLO?
YOLO (You Only Look Once) is a family of deep learning models designed for fast object Detection.
There are many versions of YOLO, we are using YOLO version 3.
The first version proposed the general architecture, where the second version refined the design and made use of predefined anchor boxes to improve the bounding box proposal, and version three further refined the model architecture and training process.
It is based on the idea that:
" A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. "

## Object detecion steps in YOLOv3
*	The input is a batch of images of shape (m, 416, 416, 3).
*	YOLO v3 passes this image to a convolutional neural network (CNN).
*	The last dimensions of the output are flattened to get an output volume of (52, 52, 45):
    *	Here, each cell of a 52 x 52 grid returns 45 numbers.
    *	45 = 3 * 15, where 5 is the number of anchor boxes per grid.
    *	15 = 5 + 10, where 5 is (pc, bx, by, bh, bw) and 10 is the number of classes we want to detect.
*	The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (pc, bx, by, bh, bw, c). If we expand c into an 10-dimensional vector, each bounding box is represented by 15 numbers.
*	Finally, we do the IoU (Intersection over Union) and Non-Max Suppression to avoid selecting overlapping boxes.

![image](https://github.com/Ahmed-Ezzat20/object-detection-YOLO-v3/assets/118759108/ce993095-03cf-4138-b1c5-0be079709d98)

## Darknet-53
Darknet-53 is a convolutional neural network that acts as a backbone for the YOLOv3 object detection approach. it is used as a feature extractor.
It is a convolutional neural network that is 53 layers deep. 
Darknet-53 mainly composed of 3 x 3 and 1 x 1 filters with skip connections like the residual network in ResNet.

![image](https://github.com/Ahmed-Ezzat20/object-detection-YOLO-v3/assets/118759108/45afb7ab-2d74-4e2a-9528-8f2490c21ea8)


## YOLOv3 architecture
*	YOLO v3 uses a variant of Darknet, which originally has 53 layer network trained on Imagenet.
*	For the task of detection, 53 more layers are stacked onto it, giving us a 106 layer fully convolutional underlying architecture for YOLO v3.
*	In YOLO v3, the detection is done by applying 1 x 1 detection kernels on feature maps of three different sizes at three different places in the network.

![image](https://github.com/Ahmed-Ezzat20/object-detection-YOLO-v3/assets/118759108/f9637fd1-4e94-414b-870d-3807b3ea78db)


*	The shape of detection kernel is 1 x 1 x (B x (5 + C)). Here B is the number of bounding boxes a cell on the feature map can predict, '5' is for the 4 bounding box attributes and one object confidence and C is the no. of classes.
In our project we can see B of each detection kernels of our three (the number bounding boxes each output layer can predict) as follows:

## How to use YOLOv3
First we need to download the model files. Weights and cfg (or configuration). files can be downloaded from this repositry.

Download the model files and put them in the same directory of the project file.

This model is pre-trained on "vegetables and fruits" dataset, Using this pre-trained weights means that you can only use YOLO for object detection with any of the 10 pre-trained classes that come with our dataset. the classes are:(apple 0-banana 1-watermelon 2-pineapple 3-orange 4-mango 5-lemon 6-kiwi 7-corn 8-tomato 9)

### Notice that:
The default network diagram for YOLOv3 model is as follows:

![image](https://github.com/Ahmed-Ezzat20/object-detection-YOLO-v3/assets/118759108/ac336d0b-3f47-42ee-9e1c-d4e68fdfa5c3)


As you can see, the input image should be in a specific size as model designed. In the image above, the input size should be 416x416 pixels. using that structure, we need specific config and weights files.





