import fitz
import os
import json
import re
import numpy as np
from PIL import Image
import requests
import cv2
import pytesseract
import imutils
from fuzzywuzzy import fuzz
# from statistics import stdev
# from elasticsearch import Elasticsearch, helpers
from yolo_detection import detect_batch_frame
# from date_detection import date_detect
import onnxruntime
import datetime
from googletrans import Translator

# from elasticsearch import Elasticsearch, helpers

# es = Elasticsearch(hosts=["http://localhost:9200"], timeout=240, max_retries=2, retry_on_timeout=True)

provider = os.getenv('PROVIDER', 'CUDAExecutionProvider')
lsq_model = onnxruntime.InferenceSession("models/yolov5/lsq.onnx", providers=[provider])
date_model = onnxruntime.InferenceSession("models/yolov5/date.onnx", providers=[provider])

translator = Translator()

def convert_img(page, zoom = 2.2):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.getPixmap(matrix = mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img
    

def sort_block(blocks):
    result = []
    for block in blocks:
        if block['type'] == 0:
            x0 = str(int(block['bbox'][0] + 0.99999)).rjust(4, "0")
            y0 = str(int(block['bbox'][1] + 0.99999)).rjust(4, "0")

            sortkey = y0 + x0
            result.append([block, sortkey])
    
    result.sort(key = lambda x: x[1], reverse=False)
    return [i[0] for i in result]


def get_text(block):
    text = ''
    for lines in block['lines']:
        for span in lines['spans']:
            text += span['text']
    return text


def save_pdf(link, save_path = 'cfs_temp.pdf'):
    if "https://dmec.moh.gov.vn" in link:
        reponse = requests.get(link, verify=False)
        with open(save_path, 'wb') as fd:
            fd.write(reponse.content)
        return save_path
    return link

def rotation_check(img):
 
    # image = cv2.imread("TuTable/000197_Appendix_001.jpg")
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
    rot_data = pytesseract.image_to_osd(img)
    # print("[OSD] "+rot_data)
    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)
    
    angle = float(rot)
    
    # rotate the image to deskew it
    rotated = imutils.rotate_bound(img, angle) #added
    
    return rotated, angle

def rotate_box(point, angle=0):

    return point

def annot_box(page,zoom):
    drawed = page.get_drawings()
    annot = page.annots()
    
    p = []

    for a in annot:
        point = a.rect
        point = point*zoom
        point = [point[0]-7, point[1]-3, point[2]+7, point[3]+3]
        p.append(point)

    for d in drawed:
        point = d['rect']
        point = point*zoom
        point = [point[0]-7, point[1]-3, point[2]+7, point[3]+3]
        p.append(point)
    
    return(p)

def pdfimage_process(pdf_path, lsq_model, date_model, check_annot=False):
    doc = fitz.open(pdf_path)
    images = []
    # points = []
    annots = []
    lsq_detect = False
    page_detect = None
    for i, page in enumerate(doc):
        print(i)
        zoom = 2.2
        list_point=[]
        if check_annot:
            p = annot_box(page, zoom)
            list_point = rotate_box(p, 0.0)
        img = convert_img(page,zoom)

        try:
            img, angle = rotation_check(img)
        except:
            img = img
            angle = 0.0
        im0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        images.append(im0)
        
        mpt_img = Image.new('RGB',(img.size[0],img.size[1]),(225,225,225))

        if check_annot:
            label = detect_batch_frame(lsq_model, [im0], image_size=(640,640))
            if len(label) > 0:
                lsq_detect = True
                page_detect = i+1
            date_result, mpt_img = detect_batch_frame(date_model, [im0], image_size=(1280,1280), mpt_img=mpt_img)
        else:
            date_result = 0
        mpt_flag = 0
        if list_point:
            for k, p in enumerate(list_point):
                p_img = img.crop(p)
                if p_img.size[1]<18:
                    continue
                mpt_flag = 1
                mpt_img.paste(p_img,(int(p[0]),int(p[1])))
            if mpt_flag or date_result:
                annots.append(cv2.cvtColor(np.array(mpt_img), cv2.COLOR_RGB2BGR))
        else:
            if date_result:
                annots.append(cv2.cvtColor(np.array(mpt_img), cv2.COLOR_RGB2BGR))
            else:
                annots.append(None)   
    return images, annots, (lsq_detect,page_detect)
    
def push_result(result, horizontal_results):
    buffer = []
    x=0
    for res in result:
        for text in res: 
            article = {"_id": x, "_index": "articles", "title": text[:5000]}
            x+=1
            buffer.append(article)
    for res in horizontal_results:
        for text in res: 
            article = {"_id": x, "_index": "articles", "title": text[:5000]}
            x+=1
            buffer.append(article)
    if buffer:
        helpers.bulk(es, buffer)
        es.indices.refresh(index='articles')
    return x



def search(query, limit=1):
    query = {
        "size": limit,
        "query": {
            "query_string": {"query": query}
        }
    }
    results = []
    for result in es.search(index="articles", body=query)["hits"]["hits"]:
        source = result["_source"]
        results.append(source["title"])
    
    return results
def del_data(idx='articles'):
    es.indices.delete(index=idx)

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return re.sub(' +', ' ',text)
    
def date_process(date):
    date = " ".join(date.split())
    try:
        mid = int(re.search(r"20", date).start())
        if mid -28 <0:
            start = 0
        else:
            start = mid-28
        
        date = date[start:]
    except:
        date = ''
    return date

def datefinder_process(date, dat):
    dates = []
    for _d in date:
        _d = _d.replace(tzinfo=None)
        if _d > datetime.timedelta(days=2000) + datetime.datetime.now() or _d < datetime.datetime.now() - datetime.timedelta(days=2000):
           continue
        year = str(_d.year)
        month = str(_d.month)
        month_name = str(_d.strftime("%b"))
        if (month in dat or month_name in dat) and year in dat:
            dates.append(_d)
    return dates
# def ranksearch(query, limit):
#   results = [text for _, text in search(query, limit * 10)]
#   return [(score, results[x]) for x, score in similarity(query, results)][:limit]

def trans(query,translator=translator):
    result = translator.translate(query, src='vi', dest='en')
    return result.text

def compare(str_1, str_2, func = fuzz.token_set_ratio, score = 50):
    if func(str_1, str_2) > score:
        return True
    return False

def normal_search(vertical_results=[],horizontal_results=[], info=None, score=60):
    results = []
    results.extend(vertical_results)
    results.extend(horizontal_results)
    for text in results:
        if compare(info.lower(), re.sub(' +', ' ',text[:-7].lower()), score=score):
            return True, text
    return False, -1
    # for key in results.keys():
    #     for _, text in results[key]["Text"]:
    #         if compare(info, text, score = 60):
    #             return True, int(key.split("_")[-1])
    # return False, -1
