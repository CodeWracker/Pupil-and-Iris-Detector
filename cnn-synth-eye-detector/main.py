import pickle
from pprint import pprint
import cv2
import numpy as np
if __name__ == "__main__":
    file = open('./processed/results.pickle', 'rb')
    data = pickle.load(file)
    images = np.array(data['image'])
    print(images.shape)
    X,Y,Ri,Rp = data['x_eye'],data['y_eye'],data['iris_radius'],data['pupil_radius']
    for i in range(0,len(images)):
        dt = images[i]
        x = X[i]
        y = Y[i]
        ri = Ri[i]
        rp = Rp[i]

        img = np.array(dt)
        print(x,y,ri,rp)
        print(img.shape)
        print()
        img = img.reshape(80,120)
        img = cv2.circle(img,(x,y),int(rp),(255, 255, 255),1)
        img = cv2.circle(img,(x,y),int(ri),(255, 255, 255),1)
        cv2.imshow("Img",img)
        cv2.waitKey(0)