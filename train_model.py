import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras.utils import plot_model


def read_images(img_dir, img_size):

    ################################################################
    # Recursively looking to the subdirectories of "img_dir"
    # This is the example of the expected file directory.
    # 1
    #    img1.png
    #    img2.png
    #      ...
    # 2
    #    img1.png
    #    img2.png
    #      ...
    # 3
    #    img1.png
    #    img2.png
    #      ...
    ################################################################

    filepaths = os.listdir(img_dir)
    total_files = [len(d) + len(f) for p, d, f in os.walk(img_dir)]
    total_class = len(filepaths)
    total_imgs = sum(total_files) - total_class

    print("Total_class: {}".format(total_class))
    print("Total_imgs: {}".format(total_imgs))

    j = 0
    imgs = []
    y = np.zeros((total_imgs, total_class))

    for i, filepath in enumerate(filepaths):
        dir_files = os.path.join(img_dir, filepath)
        files = os.listdir(dir_files)
        for j, f in enumerate(files, start=j + 1):
            img_file = os.path.join(dir_files, f)
            img = cv2.resize(cv2.imread(img_file), img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img)
            y[j - 1, i] = 1

    return imgs, y


# Plots randomly selected 20 img from the dataset
def plot_data(imgs, y):

    fig = plt.figure(figsize=(12, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        r_idx = np.random.randint(0, len(y) - 1)
        ax.imshow(np.squeeze(imgs[r_idx]), cmap='gray')
        ax.set_title(np.argmax(y[r_idx]))

    plt.show()


def get_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))

    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--plot_data", type=bool, default=False)

    args = parser.parse_args()

    # We statically defined the size of the images and the input shape of the
    # network which is (120, 120, 1)
    img_size = (120, 120)
    imgs, y = read_images(args.img_dir, img_size)

    imgs = np.array(imgs)
    imgs = np.expand_dims(imgs, axis=3)
    imgs = imgs.astype(np.float32) / 255

    # Plotting randomly selected 20 image from the dataset.
    if args.plot_data:
        plot_data(imgs, y)

    input_shape = (120, 120, 1)
    output_shape = y.shape[1]
    model = get_model(input_shape, output_shape)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, to_file="model.png")

    history = model.fit(imgs, y, epochs=args.epochs, validation_split=args.val_split, shuffle=True)

    model_name_f = args.model_name + ".h5"
    model.save(model_name_f)

    print("Model saved to : ", os.path.join(os.getcwd(), model_name_f))


if __name__ == '__main__':
    main()
