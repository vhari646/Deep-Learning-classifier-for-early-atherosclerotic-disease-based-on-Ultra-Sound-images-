# Deep-Learning-classifier-for-early-atherosclerotic-disease-based-on-Ultra-Sound-images-
Deep Learning classification of early atherosclerotic disease based on Ultra-Sound images of the carotid arterial wall tissue.

1. Set up new Anaconda virtual environment
Next, we'll work on setting up a virtual environment in Anaconda for tensorflow-gpu. 
From the Start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”.
In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:
```
C:\> conda create -n tensorflow1 pip python=3.6
```
Then, activate the environment by issuing:
```
C:\> activate tensorflow1
```
Install tensorflow-gpu in this environment by issuing:
```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```
Install the other necessary packages by issuing the following commands:
```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```

2. Configure PYTHONPATH environment variable
A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories.)
```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
Every time the "tensorflow1" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again.
Finally, run the following commands from the C:\tensorflow1\models\research directory:
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

3. Label Pictures
Label the images using LabelImg.
LabelImg saves a .xml file containing the label data for each image.
These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. 
Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.

```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python sizeChecker.py --move
```

4. Generate Training Data
With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. 
Use the xml_to_csv.py and generate_tfrecord.py scripts.

First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```
This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder. 



Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

### 5. Create Label Map and Configure Training
The last thing to do before training is to create a label map and edit the training configuration file.

The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder.
```
item {
  id: 1
  name: 'Class1'
}

item {
  id: 2
  name: 'Class2'
}

item {
  id: 3
  name: 'Class3'
}

```
The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned in Step 4, the labelmap.pbtxt file will look like:
```

### 6. Run the Training

From the \object_detection directory, issue the following command to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2.config
```
If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins.


Each step of training reports the loss. It will start high and get lower and lower as training progresses. For my training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 0.8. Train the model until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.

You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the tensorflow1 virtual environment, change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training
```
This will create a webpage on your local machine at YourPCName:6006, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.

<p align="center">
  <img src="doc/loss_graph.JPG">
</p>

The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the command prompt window. I typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

### 7. Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.

### 8. Use Your Newly Trained Object Detection Classifier!
The object detection classifier is all ready to go! 

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. 

To test your object detector, move a picture of the object or objects into the \object_detection folder, and change the IMAGE_NAME variable in the Object_detection_image.py to match the file name of the picture.

To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tensorflow1” virtual environment activated) and press ENTER. This will open IDLE, and from there, you can open any of the scripts and run them.

If everything is working properly, the object detector will initialize for about 10 seconds and then display a window showing any objects it’s detected in the image!

````
## Architecture of the Faster RCNN model
![1](https://user-images.githubusercontent.com/39281308/56237462-5472ac00-608c-11e9-975a-b8e6c06829c9.PNG)

## Implementation of the Classifier
![2](https://user-images.githubusercontent.com/39281308/56237464-5472ac00-608c-11e9-8f87-ed2a9a752759.PNG)

## Difference between Class - 1 & Class - 3
![11](https://user-images.githubusercontent.com/39281308/56238696-5be78480-608f-11e9-9142-c6e624e261fb.PNG)

## Creating the Database
# Labelling
![3](https://user-images.githubusercontent.com/39281308/56237465-5472ac00-608c-11e9-9e83-acfc372c8bb2.PNG)

# Database Split-up
![4](https://user-images.githubusercontent.com/39281308/56237466-5472ac00-608c-11e9-8e03-dbadc5699de6.PNG)
![5](https://user-images.githubusercontent.com/39281308/56237467-5472ac00-608c-11e9-8227-38c29422cde7.PNG)

## Results
# Accuracy of the Classifier
![6](https://user-images.githubusercontent.com/39281308/56237468-550b4280-608c-11e9-8b4a-675f7fc13781.PNG)

![7](https://user-images.githubusercontent.com/39281308/56237470-550b4280-608c-11e9-8761-692f8573e371.PNG)

## Class - 1 Detection
![8](https://user-images.githubusercontent.com/39281308/56237471-550b4280-608c-11e9-8c2f-97e5d055ceae.PNG)

## Class -2  Detection
![9](https://user-images.githubusercontent.com/39281308/56237472-550b4280-608c-11e9-93d8-0e340cdc0445.PNG)

## Class - 3 Detection
![10](https://user-images.githubusercontent.com/39281308/56237473-55a3d900-608c-11e9-982a-27649eb160ca.PNG)

Credits - Evan EdjeElectronics - edje.electronics@gmail.com
Licenses - # Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
