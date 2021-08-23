import yaml
import torch
from modules.easyocr.easyocr import easyocr
from modules.easyocr.vietocr import vietocr_seq2seq
from paddleocr import PaddleOCR
from modules.mrc.mrc_model import MRCQuestionAnswering
from modules.mrc.utils import *
import nltk
nltk.download('punkt')


def load_cell_detection():
    model = torch.hub.load('modules/yolov5', 'custom', path = 'modules/yolov5/weights/cell/cell_v1.pt', source = 'local')
    model.conf = 0.5
    return model

def load_table_detection():
    model = torch.hub.load('modules/yolov5', 'custom', path = 'modules/yolov5/weights/table/table_v1.pt', source = 'local')
    model.conf = 0.25
    return model

def load_easyocr(ppocr_det):
    model = easyocr.Reader(['vi','en'], verbose=False, gpu = True, paddle_ocr = ppocr_det)
    return model

def load_vietocr(ppocr_det):
    model = vietocr_seq2seq.VietOCR(config_path='modules/easyocr/vietocr/config/vgg-seq2seq-dmec.yml', paddle_ocr=ppocr_det)
    return model

def load_paddle_detection():
    model = PaddleOCR(det_model_dir='modules/easyocr/paddle/det_v2',
                      use_angle_cls=False, 
                      rec_model_dir = 'modules/easyocr/paddle/rec', cls_model_dir = 'modules/easyocr/paddle/cls',
                      lang='latin', 
                      rec_char_type='en', gpu_mem = 4000, rec = False)
    return model


class Predictor():
    
        def __init__(self, model_checkpoint):
            self.model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        def answering(self, QA_input):
            inputs = [tokenize_function(QA_input,self.tokenizer)]
            inputs_ids = data_collator(inputs,self.tokenizer)
            outputs = self.model(**inputs_ids)
            answer = extract_answer(inputs, outputs, self.tokenizer)[0]
            return answer['answer']

def load_ner():
    model_checkpoint = "modules/mrc/weights/vi-mrc-base"
    model = Predictor(model_checkpoint)
    return model