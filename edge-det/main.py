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

    def detect_iris(self):
        
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        ret, o1 = cv2.threshold(gray,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        ret, o2 = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
        ret, o3 = cv2.threshold(gray,0,255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU )
        ret, o4 = cv2.threshold(gray,0,255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU )
        ret, o5 = cv2.threshold(gray,0,255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU )
        
        '''cv2.imshow("OTSU 1", o1)
        cv2.imshow("OTSU 2", o2)
        cv2.imshow("OTSU 3", o3)
        cv2.imshow("OTSU 4", o4)
        cv2.imshow("OTSU 5", o5)'''
        
        blurred = cv2.bilateralFilter(o5,10,50,50)
        print(blurred.shape)
        



        col_count = np.zeros((290))
        row_count = []
        for i in range(blurred.shape[0]):
            aux_row = 0
            aux_col = 0
            for j in range(blurred.shape[1]):
                aux_row = aux_row + blurred[i][j]/blurred.shape[1]
                col_count[j] = col_count[j] + blurred[i][j]/blurred.shape[0]
            row_count.append(aux_row)
        cv2.imshow("Blurred", blurred)
        print((col_count.shape))
        print(np.array(row_count).shape)


        
        x_col = np.arange(0.0, blurred.shape[1], 1)
        print(x_col.shape)
        fig_c, col_ax = plt.subplots()
        col_ax.plot(x_col, col_count)
        col_ax.set(xlabel='Columns (s)', ylabel='Color Med',
            title='Column Color count')
        col_ax.grid()


        x_row = np.arange(0.0, blurred.shape[0], 1)
        fig_r, row_ax = plt.subplots()
        row_ax.plot(x_row, row_count)
        row_ax.set(xlabel='Rows (s)', ylabel='Color Med',
            title='Row Color count')
        row_ax.grid()
        plt.show()


        
        plt.show()

        minDist = 1
        param1 = 20 # 500
        param2 = 50 # 200 #smaller value-> more false circles
        minRadius = 1
        maxRadius = 300 #10

        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(self._img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print("HERE")
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
            self.detect_iris()
            cv2.imshow("Result", self._img)
            cv2.waitKey(0)
        else:
            print('Image file "' + self._img_path + '" could not be loaded.')

#id = iris_detection('c:\\temp\\eye_image.jpg')
id = iris_detection('./imgs/eye.png')
id.start_detection()