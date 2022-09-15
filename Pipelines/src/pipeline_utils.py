import cv2
import os
import mahotas 
import matplotlib.pyplot as plt    
import pandas as pd
from glob import glob
import os
import numpy as np
import argparse
import random as rng

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def imread(path):
    img =  cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img

def ingestao(img_root):
    
    data = glob(os.path.join(img_root,'**','*.jpg'))
    data = pd.DataFrame(data, columns=['img'])
    data['y_true'] = data['img'].apply(lambda x: x.split(os.sep)[-2]).astype(str)
    
    # leitura da imagem
    data['img'] = data['img'].apply(imread)
   
    return data
def rgb2gray(img, weights=[]):
    if len(weights) == 3:
       
    
        (canalVermelho, canalVerde, canalAzul) = cv2.split(img)
        out = canalVermelho*weights[0] + canalVerde*weights[1] + canalAzul*weights[2] 
        np.clip(out,a_min=0, a_max=255, out=out)
        
        return (out*255).astype(np.uint8) 
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def binarizacao_otsu(img, code = cv2.COLOR_BGR2GRAY):
    img1 = cv2.cvtColor(img, code)
    T = mahotas.thresholding.otsu(img1)
    temp = img1.copy() 
    temp[temp > T] = 255 
    temp[temp < 255] = 0 
    temp = cv2.bitwise_not(temp) 
    return temp


class Pipeline2():
    def __init__(self,
                 rgb2grayArgs = {"weights":[]},
                 bilateralFilterArgs={ 'd':3, 'sigmaColor':21,'sigmaSpace':21},
                 gaussianBlurArgs = {'ksize':(7,7),'sigmaX':0},
                 cannyArgs = {'threshold1':70, 'threshold2':200}
                 ):
        self.steps = ['rgb2gray','bilateralFilter','gaussianBlur','canny','boundingBox','final']
        self.steps_outputs = dict([(s,"") for s in self.steps])
        self.rgb2grayArgs = rgb2grayArgs
        self.bilateralFilterArgs = bilateralFilterArgs
        self.gaussianBlurArgs = gaussianBlurArgs
        self.cannyArgs = cannyArgs
    
    def __getitem__(self, step):
        return self.steps_outputs[step]
    def get_bboxes(self,canny):
        contours,_ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = []
        bboxes = []

        for i, c in enumerate(contours):
            contours_poly.append(cv2.approxPolyDP(c, 3, True))
            bbox = cv2.boundingRect(contours_poly[i])
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            bboxes.append((pt1,pt2))
            
        return  contours_poly ,  bboxes
    
    def draw_countours(self,img,countours,boxes):
        img_out = img.copy()
        for i,(c,b) in enumerate(zip(countours,boxes)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(img_out, countours, i, color)
            
            cv2.rectangle(img_out, *b, color, 2)
        return img_out
    
    def transform(self, img):
        # grayscale
        # grayscale
        self.steps_outputs['rgb2gray'] = rgb2gray(img,**self.rgb2grayArgs)
    
        # filtro bilateral
        if self.bilateralFilterArgs != {}:
            self.steps_outputs['bilateralFilter'] = cv2.bilateralFilter(self.steps_outputs['rgb2gray'],**self.bilateralFilterArgs)
        else:
            self.steps_outputs['bilateralFilter'] = self.steps_outputs['rgb2gray']
        
        #gaussian blur
        if self.gaussianBlurArgs != {}:
            self.steps_outputs['gaussianBlur'] = cv2.GaussianBlur(self.steps_outputs['bilateralFilter'],**self.gaussianBlurArgs)
        else:
            self.steps_outputs['gaussianBlur'] = self.steps_outputs['bilateralFilter']
        
        #canny
        if self.cannyArgs != {}:
            self.steps_outputs['canny'] = cv2.Canny(self.steps_outputs['gaussianBlur'], **self.cannyArgs)
        else:
            self.steps_outputs['canny'] = self.steps_outputs['gaussianBlur']
        
        
        self.steps_outputs['contours'],self.steps_outputs['bboxes'] = self.get_bboxes(self.steps_outputs['canny'])
        
        
        self.steps_outputs['final'] = self.draw_countours(img,
                                                          self.steps_outputs['contours'],
                                                          self.steps_outputs['bboxes'])
        

        return self.steps_outputs

if __name__ == "__main__":
    path = "/home/eduardo/Downloads/projetos/classificacao_plantas/abies_concolor/12995307070714.jpg"
    # img = imread(path)
    pipeline2 = Pipeline2()
    dados = ingestao('/home/eduardo/Downloads/projetos/classificacao_plantas')
    data2 = dados['img'].apply(pipeline2.transform)
    # t = p.transform(img)
    # # _otsu = binarizacao_otsu(img)
    # print(t.keys())
    i = 110
    show_image(data2.iloc[i]['rgb2gray'])
    show_image(data2.iloc[i]['bilateralFilter'])
    show_image(data2.iloc[i]['gaussianBlur'])
    show_image(t['contours'])
    show_image(p['bilateralFilter'])
    print
    