import argparse
import glob

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rgbhistogram import RGBHistogram

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--images', required=True, help='Path to the image dataset.', type=str)
argparser.add_argument('-m', '--masks', required=True, help='Path to the image dataset.', type=str)
argparser.add_argument('-s', '--sample-size', help='Number of images to classify.', default=10, type=int)
arguments = vars(argparser.parse_args())

image_paths = sorted(glob.glob(arguments['images'] + '/*.png'))
mask_paths = sorted(glob.glob(arguments['masks'] + '/*.png'))

data = []
target = []

descriptor = RGBHistogram([8, 8, 8])

for (image_path, mask_path) in zip(image_paths, mask_paths):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    features = descriptor.describe(image, mask)

    data.append(features)
    target.append(image_path.split('_')[-2])

target_names = np.unique(target)
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3, random_state=42)

model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), target_names=target_names))

for i in np.random.choice(np.arange(0, len(image_paths)), arguments['sample_size']):
    image_path = image_paths[i]
    mask_path = mask_paths[i]

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    features = descriptor.describe(image, mask)

    flower = label_encoder.inverse_transform(model.predict([features]))[0]
    print(image_path)
    print('I think this flower is a {}'.format(flower.upper()))
    cv2.imshow('Image', image)
    cv2.waitKey(0)
