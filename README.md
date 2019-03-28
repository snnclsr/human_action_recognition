## Human Action Recognition using Deep Learning (Demo Project)

In this project, we developed human action recognition system using deep learning. You can check our final model's evaluation from video below.

![VIDEO WILL BE HERE!]()

The dataset can be found [here](http://www.nada.kth.se/cvap/actions/). We used three actions from dataset which are hand waving, boxing, movement. We also created our own data which is no movement dataset which includes mostly black images.

Code is seperated into three parts.

`create_data.py`: In this function, we create the training data from the original dataset.

- `toMH()`: This function takes the `filename`, lines in sequences file which includes `filename`, the current person's number and file number as an argument. **The first line where the assignment to video_src should be fixed. We manually assign the filename to handclapping_frames.**

`motion_history.py`: In this function, we start the camera and capture frames using OpenCV. We define possible actions and load the model from `args.model_dir`. The main flow of the program:

* Calculate the absoulute difference between current and previous frames using `cv2.absdiff()`.
* Convert the absolute difference between frames to grayscale frame/image using `cv2.cvtColor()`.
* Apply threshold to grayscale image using `cv2.threshold()`.
* Update the motion history using `cv2.motempl.updateMotionHistory()`.
* Normalize the motion history using `np.clip()`.
* Show motion history and raw frames on the screen using `cv2.imshow()`.
* For prediction, we resize the current frame to (120x120) and predict the frame using `model.predict()`.
* Stack 20 consecutive prediction.
* Find the most frequent prediction and print as result using `np.bincount()` and `np.argmax()`.
* Clear the stack.


`train_model.py`: This is the main file which includes everything about training models. There are three functions in this file.

- `read_images`: this function recursively looks into the subdirectories of `img_dir` argument. Read files using OpenCV and resize it(img_size:120x120) and then convert to grayscale image.
- `plot_data`: this function plots randomly 20 images from dataset.
- `get_model`: we define the neural network architecture in this function. Model can be seen [here]().



After downloading the dataset, how to run files:

1. First of all, we should create the training dataset from the original data. For example, to create handclapping data:
`python create_data.py handclapping`

2. After creating the dataset, we should train our model. To do that:
`python train_data.py images` -- here images should be the output of `create_data.py` file.


