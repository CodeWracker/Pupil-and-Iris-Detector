import cv2
import numpy as np
import matplotlib.pyplot as plt
class iris_detection():
    def __init__(self, image_path):
        '''
        initialize the class and set the class attributes
        '''
        self._img = None
        self._img_path = image_path
        self._pupil = None

    def load_image(self):
        '''
        load the image based on the path passed to the class
        it should use the method cv2.imread to load the image
        it should also detect if the file exists
        '''
        self._img = cv2.imread(self._img_path)
        # If the image doesn't exists or is not valid then imread returns None
        if type(self._img) is type(None):
            return False
        else:
            return True
        
    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
    def detect_pupil(self):
        # Read image
        
        self.convert_to_gray_scale()
        


        ret,thresh = cv2.threshold(self._img,35,255,0)
        print(self._img.min(),self._img.max() - self._img.mean())

        
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        print(len(contours))
        if len(contours)>1:
            cnt = contours[1]
        else:
            cnt = contours[0]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(self._img,center,radius,(255,255,255),2)
        cv2.imshow("Result", self._img)
        cv2.waitKey(0)
    def start_detection(self):
        '''
        This is the main method that will be called to detect the iris
        it will call all the previous methods in the following order:
        load_image
        convert_to_gray_scale
        detect_pupil
        detect_iris
        then it should display the resulting image with the iris only
        using the method cv2.imshow
        '''
        if(self.load_image()):
            #self.convert_to_gray_scale()
            self.detect_pupil()
            
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')

#id = iris_detection('c:\\temp\\eye_image.jpg')
id = iris_detection('./imgs/eye.png')
id.start_detection()