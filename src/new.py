import math
import os
import random
from collections import Counter
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class CustomDataGenerator(Sequence):
    def __init__(self, x, y, label_encoder, batch_size=32, target_size=(128, 128),
                 rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                 zoom_range=0.2, horizontal_flip=True, test_split=0.1, train: bool = True):
        self.label_encoder = label_encoder
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.test_split = test_split
        self.image_files = []
        self.labels = []
        self.classes = set(item.split("_")[0] for item in os.listdir(dataset_path))
        self.train = train

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_image_files_path = self.x[start_index:end_index]
        batch_labels = self.y[start_index:end_index]

        batch_images = []
        for image_path in batch_image_files_path:
            image = Image.open(image_path)
            image = image.resize(self.target_size)
            # Convert the image to numpy array
            image_array = np.array(image)
            # Apply data augmentation
            if self.train:
                image_array = self.apply_augmentation(image_array)
            # Normalize the image data
            image_array = image_array / 255.0
            batch_images.append(image_array)

        # One-hot encode labels
        batch_labels_encoded = self.label_encoder.transform(batch_labels)
        batch_labels_one_hot = to_categorical(batch_labels_encoded, num_classes=len(self.classes))

        return np.array(batch_images), np.array(batch_labels_one_hot)

    def apply_augmentation(self, image_array):
        # Apply data augmentation
        datagen = ImageDataGenerator(rotation_range=self.rotation_range,
                                     width_shift_range=self.width_shift_range,
                                     height_shift_range=self.height_shift_range,
                                     zoom_range=self.zoom_range,
                                     horizontal_flip=self.horizontal_flip)
        image_array = datagen.random_transform(image_array)
        return image_array

    def on_epoch_end(self):
        # np.random.shuffle(self.train_files)
        range_items = list(range(len(self.x)))
        random.shuffle(range_items)
        self.x = np.array(self.x)[range_items].tolist()
        self.y = np.array(self.y)[range_items].tolist()


    # def get_train_data(self):
    #     train_images = []
    #     train_labels = []
    #     for image_file in self.train_files:
    #         image_path = os.path.join(self.dataset_path, image_file)
    #         image = Image.open(image_path)
    #         image = image.resize(self.target_size)
    #         image_array = np.array(image) / 255.0
    #         train_images.append(image_array)
    #         train_labels.append(self.labels[self.image_files.index(image_file)])
    #
    #     # One-hot encode train labels
    #     train_labels_encoded = self.label_encoder.transform(train_labels)
    #     train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=len(self.classes))
    #
    #     return np.array(train_images), np.array(train_labels_one_hot)

    # def get_test_data(self):
    #     test_images = []
    #     test_labels = []
    #     for image_file in self.test_files:
    #         image_path = os.path.join(self.dataset_path, image_file)
    #         image = Image.open(image_path)
    #         image = image.resize(self.target_size)
    #         image_array = np.array(image) / 255.0
    #         test_images.append(image_array)
    #         test_labels.append(self.labels[self.image_files.index(image_file)])
    #
    #     # One-hot encode test labels
    #     test_labels_encoded = self.label_encoder.transform(test_labels)
    #     test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=len(self.classes))
    #
    #     return np.array(test_images), np.array(test_labels_one_hot)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), strides=2),
        # layers.Dropout(0.5),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', strides=2),
        # layers.Dropout(0.5),
        layers.Conv2D(124, (3, 3), activation='relu', strides=2),
        # layers.Dropout(0.5),
        # layers.Conv2D(256, (3, 3), activation='relu', strides=2),
        # layers.Conv2D(512, (3, 3), activation='relu', strides=2),
        # layers.Dropout(0.5),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(256, (3, 3), activation='relu', strides=2),
        # layers.Dropout(0.5),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # layers.Dropout(0.5),
        # layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(4, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_and_save_curves(history, filename="curves.png"):
    """
    Plots and saves separate loss and accuracy curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss and accuracy on the same figure

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.savefig(filename.replace(".png", "_loss.png"))
    plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

    # Save the figure
    plt.savefig(filename.replace(".png", "_accuracy.png"))
    plt.close()


def dataset_split(dataset_path: str, test_split: float = 0.2, val_split: float = 0.2):
    images_files = []
    labels = []
    for item in os.listdir(dataset_path):
        item_path = join(dataset_path, item)
        images_files.append(item_path)
        labels.append(item.split("_")[0])

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(images_files,
                                                        labels,
                                                        test_size=test_split,
                                                        random_state=42,
                                                        stratify=labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=val_split,
                                                      random_state=42,
                                                      stratify=y_train)
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    # Create CustomDataGenerator for training data
    dataset_path = "../dataset"
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_split(dataset_path)
    print("train:", Counter(y_train))
    print("val:", Counter(y_val))
    print("test:", Counter(y_test))
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_val + y_test)

    train_generator = CustomDataGenerator(x_train, y_train,
                                          label_encoder,
                                          train=True,
                                          batch_size=16,
                                          target_size=(128, 128),
                                          rotation_range=10,
                                          # width_shift_range=0.2,
                                          # height_shift_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=False)

    # Create CustomDataGenerator for validation data
    val_generator = CustomDataGenerator(x_val, y_val, label_encoder,
                                        train=False, batch_size=32, target_size=(128, 128))
    test_generator = CustomDataGenerator(x_test, y_test, label_encoder,
                                         train=False, batch_size=32, target_size=(128, 128))

    # Create the model
    model = create_model()
    print(model.summary())
    # Train the model
    history = model.fit(train_generator,
                        epochs=500,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator))

    # Evaluate the model on the test set
    print("Evaluate on test data:")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(test_loss, test_accuracy)
    # Get the original class labels
    # test_labels = [train_generator.labels[train_generator.image_files.index(file)] for file in
    #                train_generator.test_files]
    #
    # # Count the number of pictures in each class
    # class_counts = {}
    # for label in train_generator.classes:
    #     class_counts[label] = test_labels.count(label)
    #
    # print("Number of pictures in each class in the test data:")
    # for label, count in class_counts.items():
    #     print(f"{label}: {count}")
    #
    # # Evaluate the model on the training set
    # print("Evaluate on train data:")
    # train_loss, train_accuracy = model.evaluate(train_generator)
    #
    # # Get the original class labels for the training data
    # train_labels = [train_generator.labels[train_generator.image_files.index(file)] for file in
    #                 train_generator.train_files]
    #
    # # Count the number of pictures in each class
    # class_counts_train = {}
    # for label in train_generator.classes:
    #     class_counts_train[label] = train_labels.count(label)
    #
    # print("Number of pictures in each class in the training data:")
    # for label, count in class_counts_train.items():
    #     print(f"{label}: {count}")
    #
    # # Determine the number of train and validation data
    # num_train_data = len(train_generator.image_files)
    # num_val_data = len(val_generator.image_files)
    #
    # print(f"Number of train data: {num_train_data}")
    # print(f"Number of validation data: {num_val_data}")
    #
    # # Plot and save the curves
    # plot_and_save_curves(history, filename="curves.png")
