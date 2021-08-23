import yaml
import torch
from modules.easyocr.easyocr import easyocr
from modules.easyocr.vietocr import vietocr_seq2seq
from paddleocr import PaddleOCR
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# import nltk
# nltk.download('punkt')


def load_easyocr(ppocr_det):
    model = easyocr.Reader(['vi','en'], verbose=False, gpu = True, paddle_ocr = ppocr_det)
    return model

def load_vietocr(ppocr_det):
    model = vietocr_seq2seq.VietOCR(config_path='/content/drive/MyDrive/Vinbrain/DMEC/CFS_v2/ocr/modules/easyocr/vietocr/config/vgg-seq2seq-dmec.yml', paddle_ocr=ppocr_det)
    return model

def load_paddle_detection():
    model = PaddleOCR(det_model_dir='/content/drive/MyDrive/Vinbrain/DMEC/CFS_v2/ocr/modules/easyocr/paddle/det_v2',
                      use_angle_cls=False, 
                      rec_model_dir = '/content/drive/MyDrive/Vinbrain/DMEC/CFS_v2/ocr/modules/easyocr/paddle/rec', cls_model_dir = '/content/drive/MyDrive/VinBrain/DMEC/Demo/CFS_v2/ocr/modules/easyocr/paddle/cls',
                      lang='latin',
                      gpu = True, 
                      rec_char_type='en', gpu_mem = 4000, rec = False)
    return model

