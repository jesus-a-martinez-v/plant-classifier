import cv2


class RGBHistogram(object):
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image, mask=None):
        histogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)

        return histogram.flatten()
