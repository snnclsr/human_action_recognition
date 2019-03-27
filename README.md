## Human Action Recognition using Deep Learning (Demo Project)

In this project, we developed human action recognition system using deep learning. You can check our final models evaluation from video below.

![VIDEO WILL BE HERE!]()

The dataset can be found [here](http://www.nada.kth.se/cvap/actions/). We used three actions from dataset which are hand waving, boxing, movement. We also created our own data which is no movement dataset which consists mostly black images.

Code is seperated into three parts.

`create_data.py`:

`motion_history.py`:

* `train_model.py`: This is the main file which consists everything about training models. There are three functions in this file.

- `read_images`: this function recursively looks into the subdirectories of `img_dir` argument. Read files using OpenCV and resize it(img_size:120x120) and then convert to grayscale image.
- `plot_data`: this function plots randomly 20 images from dataset.
- `get_model`: we define the neural network architecture in this function. Model can be seen [here]().


