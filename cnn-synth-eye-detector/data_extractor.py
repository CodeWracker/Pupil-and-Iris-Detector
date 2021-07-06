import pickle
from pprint import pprint
import numpy as np
import cv2
import math
import os
import pandas as pd
from tqdm import tqdm
def get_image_data(path_plk,path_image):

    # open a file, where you stored the pickled data
    file = open(f'./data/{path_plk}', 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    img = cv2.imread(f'./data/{str(path_image)}')

    points_n = len(data['ldmks']['ldmks_iris_2d'])


    (x_iris,y_iris,r_iris) = 0,0,0
    for dt in data['ldmks']['ldmks_iris_2d']:
        x,y = dt
        x = int(x)
        y = int(y)
        x_iris+=x
        y_iris+=y

    x_iris = int(x_iris/points_n) 
    y_iris = int(y_iris/points_n) 
    r_iris = math.sqrt((int(data['ldmks']['ldmks_iris_2d'][0][0]) - x_iris)**2)
    r_pupil = 0
    r_pupil = math.sqrt((int(data['ldmks']['ldmks_pupil_2d'][0][0]) - x_iris)**2)  

    df = pd.DataFrame()
    df["x_eye"] = np.array([x_iris])
    df["y_eye"] = np.array([y_iris])
    df["pupil_radius"] = np.array([r_pupil])
    df["iris_radius"] = np.array([r_iris])
    df['file_name'] = np.array([ path_image.split('.png')[0] ])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img.reshape(9600,))
    return x_iris,y_iris,r_pupil,r_iris,path_image.split('.png')[0],img

def save_df(x_iris,y_iris,r_pupil,r_iris,path_image):
    df = pd.DataFrame()
    df["x_eye"] = np.array(x_iris)
    df["y_eye"] = np.array(y_iris)
    df["pupil_radius"] = np.array(r_pupil)
    df["iris_radius"] = np.array(r_iris)
    df['file_name'] = np.array(path_image )
    df.to_csv("./processed/results.csv")
    

if __name__ == "__main__":
    files = os.listdir('./data')
    #print(files)
    i = 0
    x_iris,y_iris,r_pupil,r_iris,path_image,imgs = [],[],[],[],[],[]
    for i in tqdm(range(0,len(files)-1 ,2)):
        #print(files[i+1],files[i])
        x,y,rp,ri,pi,img = get_image_data(files[i],files[i+1])
        x_iris.append(x)
        y_iris.append(y)
        r_pupil.append(rp)
        r_iris.append(ri)
        path_image.append(pi)
        imgs.append(img)
        save_df(x_iris,y_iris,r_pupil,r_iris,path_image)
    
        dt = {
            "x_eye": x_iris ,
            "y_eye":  y_iris,
            "pupil_radius":  r_pupil ,
            "iris_radius":r_iris,
            "file_name": path_image,
            "image": imgs }

        with open('./processed/results.pickle', 'wb') as handle:
            pickle.dump(dt, handle, protocol=pickle.HIGHEST_PROTOCOL)