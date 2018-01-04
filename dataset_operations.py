import os
import glob
import numpy as np
import cv2 as opencv
from sklearn.utils import shuffle
from dataset import DataSet


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for data_class in classes:
        index = classes.index(data_class)
        print('Loading {} files (Index: {})'.format(data_class, index))
        path = os.path.join(train_path, data_class, '*')
        files = glob.glob(path)
        for image_file in files:
            image = opencv.imread(image_file)
            image = opencv.resize(image, (image_size, image_size), opencv.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            complete_filename = os.path.basename(image_file)
            ids.append(complete_filename)
            cls.append(data_class)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size, classes):
    for class_name in classes:
        path = os.path.join(test_path, class_name, '*')
        files = sorted(glob.glob(path))
        x_test = []
        x_test_id = []
        print("Reading test data set of images")
        for data_file in files:
            file_base = os.path.basename(data_file)
            print(data_file)
            img = opencv.imread(data_file)
            img = opencv.resize(img, (image_size, image_size), opencv.INTER_LINEAR)
            x_test.append(img)
            x_test_id.append(file_base)
        x_test = np.array(x_test, dtype=np.uint8)
        x_test = x_test.astype('float32')
        x_test = x_test / 255
    return x_test, x_test_id


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):  # similar to datasets dict.
        pass
    data_sets = DataSets()
    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]
    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)
    return data_sets


def read_test_set(test_path, image_size, classes):
    images, ids = load_test(test_path, image_size, classes)
    return images, ids
