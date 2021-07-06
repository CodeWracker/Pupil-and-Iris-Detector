import pickle
from pprint import pprint
import cv2
import numpy as np
if __name__ == "__main__":
    file = open('./processed/results.pickle', 'rb')
    data = pickle.load(file)
    for dt in data['image']:
        img = np.array(dt)
        cv2.imshow("Img",img.reshape(80,120))
        cv2.waitKey(0)