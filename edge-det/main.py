import cv2
import numpy as np
import matplotlib.pyplot as plt
class iris_detection():
    def __init__(self, image_path):
        '''
        initialize the class and set the class attributes
        '''
        self._img = None
        self._img_out = None
        self._img_path = image_path
        self._pupil = None

    def load_image(self):
        '''
        load the image based on the path passed to the class
        it should use the method cv2.imread to load the image
        it should also detect if the file exists
        '''
        self._img = cv2.imread(self._img_path)
        self._img_out = self._img
        # If the image doesn't exists or is not valid then imread returns None
        if type(self._img) is type(None):
            return False
        else:
            return True
        
    def convert_to_gray_scale(self):
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

    def detect_iris(self):
        
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        ret, o5 = cv2.threshold(gray,0,255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU )
        blurred = cv2.bilateralFilter(o5,10,50,50)
        print(blurred.shape)
        ret, bin_img = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        col_count = np.zeros((bin_img.shape[1]))
        row_count = []
        for i in range(bin_img.shape[0]):
            aux_row = 0
            aux_col = 0
            for j in range(bin_img.shape[1]):
                aux_row = aux_row + bin_img[i][j]/bin_img.shape[1]
                col_count[j] = col_count[j] + bin_img[i][j]/bin_img.shape[0]
            row_count.append(aux_row)
        for i in range(len(row_count)):
            row_count[i] =( 255 -row_count[i])
        for i in range(len(col_count)):
            col_count[i] = (255 -col_count[i])
        
        row_count = np.array(row_count)
        row_count = row_count - row_count.min()
        media_r = 0
        for i,row in enumerate(row_count):
            media_r+= row*i/row_count.sum()
        desvio_r = 0
        for i,row in enumerate(row_count):
            if row>0:
                desvio_r = media_r - i
                break
        
        col_count = col_count - col_count.min()
        media_c = 0
        for i,col in enumerate(col_count):
            media_c+= col*i/col_count.sum()
        desvio_c = 0
        for i,col in enumerate(col_count):
            if col>(0.45*col_count.max()):
                desvio_c = media_c - i
                break
        print("x: ",media_c)
        print("y: ",media_r)
        print("raio: ",desvio_c,desvio_r)

        #cv2.putText(self._img,'+',(int(media_c),int(media_r)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        cv2.ellipse(self._img_out, (int(media_c),int(media_r)), (int(desvio_c),int(desvio_c)),0,0,360, (0, 0, 255), 2)


    def test(self):
        self.convert_to_gray_scale()
        new_img = np.zeros(self._img.shape)
        min_val = self._img.min() + 20
        print(new_img.shape)
        print(min_val)
        for i in range(0,new_img.shape[0]):
            for j in range(0,new_img.shape[1]):
                if(self._img[i][j]>min_val):
                    new_img[i][j] = 255 
        cv2.imshow("test",new_img)
        cv2.waitKey(0)
    def detect_pupil(self):
        # Read image
        
        self.convert_to_gray_scale()
        

        new_img = np.zeros(self._img.shape, dtype=self._img.dtype)
        min_val = self._img.min() + 40
        print(new_img.shape)
        print(min_val)
        for i in range(0,new_img.shape[0]):
            for j in range(0,new_img.shape[1]):
                if(self._img[i][j]>min_val):
                    new_img[i][j] = 255 
        new_img = np.array(new_img)
        
        kernel = np.ones((5,5),np.uint8)
        cv2.imshow("d",new_img)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("MORPH_OPEN",new_img)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("MORPH_CLOSE",new_img)
        cv2.waitKey(0)
        contours,hierarchy = cv2.findContours(new_img, 1, 2)
        print(len(contours))
        cnt = contours[0]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        print(x,y)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(self._img_out,center,radius,(255,255,255),2)
        

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
            #self.detect_iris()
            self.detect_pupil()
            cv2.imshow("Result", self._img_out)
            cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')

#id = iris_detection('c:\\temp\\eye_image.jpg')
id = iris_detection('./imgs/eye.png')
id.start_detection()