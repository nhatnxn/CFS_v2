import numpy as np
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)

from tools.load_model import  load_paddle_detection, load_vietocr
import cv2
import time
import re
from ocr.utils import vertical_cluster, horizontal_cluster

class Reader():
    def __init__(self):
        self.text_detector  = load_paddle_detection()
        self.ocr            = load_vietocr(self.text_detector)

    def readtext(self, img, paragraph=False, cluster=False, page=0):
        box_texts = self.text_detector.ocr(img, rec=False, cls=False)
        boxes = np.array(box_texts, dtype=int)
        try:
            top_lefts = np.stack([boxes[:, :, 0].min(axis=1), boxes[:, :, 1].min(axis=1)], axis=1)
            bot_rights = np.stack([boxes[:, :, 0].max(axis=1), boxes[:, :, 1].max(axis=1)], axis=1)
            texts = np.concatenate([top_lefts, bot_rights], axis=1).tolist()
        except:
            return [], []
        texts_ocr_poly = [[(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])] for box in texts]
        texts_ocr_predict = self.ocr.recognize(img, free_list=texts_ocr_poly, horizontal_list=[], reformat=True)

        if cluster:
            vertical_blocks = vertical_cluster(box_texts)
            horizontal_blocks = horizontal_cluster(box_texts)
            vertical_result = []
            for p in vertical_blocks:
                # print(p)
                par = ''
                if len(p)==0:
                    continue
                for cnt in p:
                    nstr = re.sub(r"'",r"",texts_ocr_predict[cnt][1])
                    nstr = re.sub(r'"',r'',nstr)
                    # print(nstr)
                    par = str(par) + " " + nstr
                par = par + ' ' + 'page {'+f'{page}'+'}'

                vertical_result.append(par)
            
            # print(horizontal_blocks)
            horizontal_result=[]
            for p in horizontal_blocks:
                par = ""
                if len(p)==0:
                    continue
                for cnt in p:
                    nstr = re.sub(r"'",r"",texts_ocr_predict[cnt][1])
                    nstr = re.sub(r'"',r'',nstr)
                    par = str(par) + " " + nstr
                par = par + ' ' + 'page {'+f'{page}'+'}'
                horizontal_result.append(par)
            return vertical_result, horizontal_result
        else:
            vertical_result = [text[1] + ' ' + 'page {'+f'{page}'+'}' for text in texts_ocr_predict]
            horizontal_result = None
            return vertical_result, horizontal_result

if __name__ == "__main__":
    text_detector  = load_paddle_detection()
    ocr            = load_vietocr(text_detector)

    img_path = 'demo/visualize.jpg'
    img = cv2.imread(img_path)
    
    
    print("Inference text detect")
    t1 = time.time()
    box_texts = text_detector.ocr(img_path, rec=False, cls=False)
    print("Time text detect: ", time.time() - t1)
    boxes = np.array(box_texts, dtype=int)
    top_lefts = np.stack([boxes[:, :, 0].min(axis=1), boxes[:, :, 1].min(axis=1)], axis=1)
    bot_rights = np.stack([boxes[:, :, 0].max(axis=1), boxes[:, :, 1].max(axis=1)], axis=1)

    texts = np.concatenate([top_lefts, bot_rights], axis=1).tolist()
    print(texts)
    
    print("Inference OCR")
    t1 = time.time()
    texts_ocr_poly = [[(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])] for box in texts]
    texts_ocr_predict = ocr.recognize(img, free_list=texts_ocr_poly, horizontal_list=[], reformat=True)
    print("Time ocr: ", time.time() - t1)
    print(texts_ocr_predict)
