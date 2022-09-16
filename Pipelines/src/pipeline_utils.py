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
from skimage.filters import threshold_otsu
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
    data = pd.DataFrame(data, columns=['img_path'])
    data['y_true'] = data['img_path'].apply(lambda x: x.split(os.sep)[-2]).astype(str)
    data['img_name'] = data['img_path'].apply(lambda x: x.split(os.sep)[-1]).astype(str)
    
    # leitura da imagem
    data['img'] = data['img_path'].apply(imread)
    return data.drop('img_path',axis=1)
def rgb2gray(img, weights=[]):
    if len(weights) == 3:
       
    
        (canalVermelho, canalVerde, canalAzul) = cv2.split(img)
        out = canalVermelho*weights[0] + canalVerde*weights[1] + canalAzul*weights[2] 
        np.clip(out,a_min=0, a_max=255, out=out)
        
        return out.astype(np.uint8) 
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
def sample_by_class(data, samples_per_class = -1):
  # data_aux = data.copy()
  classes = data['y_true'].unique()
  
  data_aux = [data.loc[data['y_true'] == c].sample(samples_per_class) for c in classes]
  data_aux = pd.concat(data_aux).reset_index(drop=True)

  return data_aux
def imshow_subplot(img,formato,c,title='',loc='left',fontsize=20,cmap='gray'):
  plt.subplot(*formato,c)
  plt.title(title,loc=loc,fontsize=fontsize)
  plt.axis('off')
  plt.imshow(img,cmap=cmap)
def run_pipeline(data, transform = None, steps =[],show_img=True,cmaps = []):
            #  save_path='.'):
 
  formato = (data.shape[0], len(steps) + 1)

  size_y = 200 
  size_x = size_y //formato[1]

  size = (size_x ,size_y)
  plt.rcParams["figure.figsize"] = size

  fontsize=18
  for i in range(formato[0]):
    c = 1
    img_original = data.loc[i,'img']
    imshow_subplot(img_original,formato=formato,c=c,title=data.loc[i,'y_true'],loc='left',fontsize=fontsize)
    c+=1
    t = transform(img_original)
    for step,cmap in zip(steps,cmaps):
      img_step = t[step]
      imshow_subplot(img_step,formato=formato,c=c,title=step,loc='left',fontsize=fontsize,cmap=cmap)

      c+=1
   
    if show_img:
      plt.show()
class Pipeline1():
    def __init__(self,
                 kmeansArgs = {'K':10,
                               'bestLabels':None,
                               # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
                               'criteria':(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85),
                               'attempts':10,
                               'flags':cv2.KMEANS_RANDOM_CENTERS},
                 customMaskArgs={'h_lower':0.0,
                                 'h_upper':1,
                                 's_lower':0.,
                                 's_upper':1.0,
                                 'v_lower':0,
                                 'v_upper':1.0}
                 ):
        self.steps = ['kmeans','rgb2hsv','customMask','otsuFilter','final']
        self.kmeansArgs = kmeansArgs
        self.customMaskArgs = customMaskArgs
       
    def kmeans(self,img,kmeansArgs):
        
        pixel_vals = img.reshape((-1,3))
        pixel_vals = np.float32(pixel_vals)
        retval, labels, centers = cv2.kmeans(pixel_vals, **kmeansArgs)
        
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        
        seg_img = segmented_data.reshape((img.shape))
        return seg_img

    def custom_mask(self,hsvImg,customMaskArgs):
        h_mask_l = hsvImg[:,:,0] >= (customMaskArgs['h_lower'] * 255)
        h_mask_u = hsvImg[:,:,0] <= (customMaskArgs['h_upper'] * 255)
        s_mask_l = hsvImg[:,:,1] >= (customMaskArgs['s_lower'] * 255)
        s_mask_u = hsvImg[:,:,1] <= (customMaskArgs['s_upper'] * 255)
        v_mask_l = hsvImg[:,:,2] >= (customMaskArgs['v_lower'] * 255)
        v_mask_u = hsvImg[:,:,2] <= (customMaskArgs['v_upper'] * 255)
        mask = h_mask_l*h_mask_u*s_mask_l*s_mask_u*v_mask_l*v_mask_u
        print(mask.max())
        return np.stack([mask]*3, axis=2) * hsvImg
    
    def otsu_filter(self, hsvImg):
        imgAux = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB)
        imgAux = cv2.cvtColor(hsvImg, cv2.COLOR_RGB2GRAY)
        lt = threshold_otsu(imgAux)

        return imgAux > lt
    
    def transform(self, img):
        output = {}
        # kmeans
        output['kmeans'] = self.kmeans(img,self.kmeansArgs)

        # hsv
        output['hsv'] = cv2.cvtColor(output['kmeans'], cv2.COLOR_RGB2HSV)
        
        # mask
        output['customMask'] = self.custom_mask(output['hsv'],self.customMaskArgs)
        
        #otsu
        output['otsuFilter'] = self.otsu_filter(output['customMask'])
        
        
        # output['contours'],output['bboxes'] = self.get_bboxes(output['canny'])
        
        output['final'] = np.stack([output['otsuFilter']]*3, axis=2) * img

        

        return output
class Pipeline2():
    def __init__(self,
                 rgb2grayArgs = {"weights":[]},
                 bilateralFilterArgs={ 'd':3, 'sigmaColor':21,'sigmaSpace':21},
                 gaussianBlurArgs = {'ksize':(7,7),'sigmaX':0},
                 cannyArgs = {'threshold1':70, 'threshold2':200}
                 ):
        self.steps = ['rgb2gray','bilateralFilter','gaussianBlur','canny','boundingBox','final']
        self.rgb2grayArgs = rgb2grayArgs
        self.bilateralFilterArgs = bilateralFilterArgs
        self.gaussianBlurArgs = gaussianBlurArgs
        self.cannyArgs = cannyArgs
    
    
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
        output = {}
        # grayscale
        output['rgb2gray'] = rgb2gray(img,**self.rgb2grayArgs)
    
        # filtro bilateral
        if self.bilateralFilterArgs != {}:
            output['bilateralFilter'] = cv2.bilateralFilter(output['rgb2gray'],**self.bilateralFilterArgs)
        else:
            output['bilateralFilter'] = output['rgb2gray']
        
        #gaussian blur
        if self.gaussianBlurArgs != {}:
            output['gaussianBlur'] = cv2.GaussianBlur(output['bilateralFilter'],**self.gaussianBlurArgs)
        else:
            output['gaussianBlur'] = output['bilateralFilter']
        
        #canny
        if self.cannyArgs != {}:
            output['canny'] = cv2.Canny(output['gaussianBlur'], **self.cannyArgs)
        else:
            output['canny'] = output['gaussianBlur']
        
        
        output['contours'],output['bboxes'] = self.get_bboxes(output['canny'])
        
        
        output['final'] = self.draw_countours(img,
                                                          output['contours'],
                                                          output['bboxes'])
        

        return output

if __name__ == "__main__":
    path = "/home/eduardo/Downloads/projetos/classificacao_plantas/abies_concolor/12995307070714.jpg"
   
    kmeansArgs = {'K':10,
                  'bestLabels':None,
                  # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
                  'criteria':(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85),
                  'attempts':10,
                  'flags':cv2.KMEANS_RANDOM_CENTERS}
    pipeline1 = Pipeline1(kmeansArgs = kmeansArgs)
    dados = ingestao('/home/eduardo/Downloads/projetos/classificacao_plantas')
    amelanchier_canadensis = glob("/home/eduardo/Downloads/projetos/classificacao_plantas/amelanchier_canadensis/*.jpg")
    img = imread(amelanchier_canadensis[0])
    i = 110
    out = pipeline1.transform(img)
    show_image(img)
    show_image(out['kmeans'])
    show_image(out['otsuFilter'])
    show_image(out['final'])
    print
    