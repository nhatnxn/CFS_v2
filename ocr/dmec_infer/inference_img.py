import numpy as np
from tools.load_model import  load_paddle_detection, load_vietocr
import cv2

if __name__ == "__main__":
    text_detector  = load_paddle_detection()
    ocr            = load_vietocr(text_detector)

    img_path = 'demo/visualize.jpg'
    img = cv2.imread(img_path)
    box_texts = text_detector.ocr(img_path, rec=False, cls=False)
    boxes = np.array(box_texts, dtype=int)
    top_lefts = np.stack([boxes[:, :, 0].min(axis=1), boxes[:, :, 1].min(axis=1)], axis=1)
    bot_rights = np.stack([boxes[:, :, 0].max(axis=1), boxes[:, :, 1].max(axis=1)], axis=1)

    texts = np.concatenate([top_lefts, bot_rights], axis=1).tolist()
    print(texts)

    texts_ocr_poly = [[(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])] for box in texts]
    texts_ocr_predict = ocr.recognize(img, free_list=texts_ocr_poly, horizontal_list=[], reformat=True)
    print(texts_ocr_predict)
