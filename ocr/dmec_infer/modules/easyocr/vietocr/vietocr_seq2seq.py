import glob
import time
import cv2
from .vietocr.tool.predictor import Predictor
from .vietocr.tool.config import Cfg
from .vietocr.tool.utils import calculate_ratio, compute_ratio_and_resize, four_point_transform, get_paragraph, reformat_input
import numpy as np
import math

class VietOCR():
    def __init__(self, config_path, paddle_ocr=None):
        # config_path = "config/vgg-seq2seq-dmec.yml"
        self.config = Cfg.load_config_from_file(config_path)
        self.predictor = Predictor(self.config)
        self.paddle_ocr = paddle_ocr


    #ocr.recognize(image, free_list = texts_ocr_poly, horizontal_list = [], reformat=True)
    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None, \
                  decoder='greedy', beamWidth=5, batch_size=1, \
                  workers=0, allowlist=None, blocklist=None, detail=1, \
                  rotation_info=None, paragraph=False, \
                  contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, \
                  y_ths=0.5, x_ths=1.0, reformat=True, output_format='standard'):

        '''

        :param img_cv_grey: IMPORTANT: cv2 image color (not grayscale)
        :param free_list: [(tl, tr, br, bl), (tl, tr, br, bl), ...] : tl ~  top left, ...
        :return:
        '''
        imgH = self.config['dataset']['image_height']
        ignore_char = None
        character = None
        recognizer = None
        converter = None
        device = None

        image_list, max_width = self.get_image_list(horizontal_list, free_list, img_cv_grey, model_height=imgH)

        result = self.get_text(character, imgH, max_width, recognizer, converter, image_list, \
                          ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths, \
                          workers, device)


        if paragraph:
            direction_mode = 'ltr'
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode=direction_mode)
        return  result

    def get_image_list(self, horizontal_list, free_list, img, model_height = 64, sort_output = True):
        '''

        :param horizontal_list:
        :param free_list:
        :param img:  cv2 color image
        :param model_height:
        :param sort_output:
        :return:
        '''
        image_list = []

        for box in free_list:
            rect = np.array(box, dtype="float32")
            transformed_img = four_point_transform(img, rect)
            image_list.append((box, transformed_img))

        #     ratio = calculate_ratio(transformed_img.shape[2], transformed_img.shape[1])
        #     new_width = int(model_height * ratio)
        #
        #     if new_width == 0:
        #         pass
        #     else:
        #         crop_img, ratio = compute_ratio_and_resize(transformed_img, transformed_img.shape[2],
        #                                                    transformed_img.shape[1], model_height)
        #         image_list.append((box, crop_img))  # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        #         max_ratio_free = max(ratio, max_ratio_free)
        #
        # max_ratio_free = math.ceil(max_ratio_free)

        # for box in horizontal_list:
        #     x_min = max(0, box[0])
        #     x_max = min(box[1], maximum_x)
        #     y_min = max(0, box[2])
        #     y_max = min(box[3], maximum_y)
        #     crop_img = img[y_min: y_max, x_min:x_max]
        #     width = x_max - x_min
        #     height = y_max - y_min
        #     ratio = calculate_ratio(width, height)
        #     new_width = int(model_height * ratio)
        #     if new_width == 0:
        #         pass
        #     else:
        #         crop_img, ratio = compute_ratio_and_resize(crop_img, width, height, model_height)
        #         image_list.append(([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img))
        #         max_ratio_hori = max(ratio, max_ratio_hori)
        #

        # max_ratio_hori = math.ceil(max_ratio_hori)
        # max_ratio = max(max_ratio_hori, max_ratio_free)
        # max_width = math.ceil(max_ratio) * model_height
        #
        # if sort_output:
        #     image_list = sorted(image_list, key=lambda item: item[0][0][1])  # sort by vertical position

        max_width = None
        return image_list, max_width

    def get_text(self, character, imgH, imgW, recognizer, converter, image_list,\
             ignore_char = '',decoder = 'greedy', beamWidth =5, batch_size=1, contrast_ths=0.1,\
             adjust_contrast=0.5, filter_ths = 0.003, workers = 1, device = 'cpu'):


        coord = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]

        result_ocr = self.predictor.batch_predict(img_list)

        result = []
        for i, zipped in enumerate(zip(coord, result_ocr)):
            box, pred = zipped
            result.append((box, pred, None))


        return result

    def readtext(self, image, decoder='greedy', beamWidth=5, batch_size=1, \
                     workers=0, allowlist=None, blocklist=None, detail=1, \
                     rotation_info=None, paragraph=False, min_size=20, \
                     contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, \
                     text_threshold=0.7, low_text=0.4, link_threshold=0.4, \
                     canvas_size=2560, mag_ratio=1., \
                     slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
                     width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, output_format='standard'):
            '''
            Parameters:
            image: file path or numpy-array or a byte stream object
            '''
            img, img_cv_grey = reformat_input(image)

            # get the 1st result from hor & free list as self.detect returns a list of depth 3

            ##### for CRAF detection
            #         horizontal_list, free_list = horizontal_list[0], free_list[0]

            ##### for CRAF detection
            horizontal_list = []

            free_list = self.paddle_ocr.ocr(img, rec=False, cls=True)

            result = self.recognize(image, horizontal_list, free_list, \
                                    decoder, beamWidth, batch_size, \
                                    workers, allowlist, blocklist, detail, rotation_info, \
                                    paragraph, contrast_ths, adjust_contrast, \
                                    filter_ths, y_ths, x_ths, False, output_format)

            return result




if __name__ == "__main__":
    config = Cfg.load_config_from_file("config/vgg-seq2seq-dmec.yml")
    predictor = Predictor(config)
    # img_path = "img_test/2_ISO_page_206_0000024.png"
    # img = Image.open(img_path)
    img_paths = sorted(glob.glob("img_test/*"))
    print(img_paths)
    # imgs = [Image.open(img_path) for img_path in img_paths]
    imgs = [cv2.imread(img_path) for img_path in img_paths]
    res = predictor.batch_predict(imgs)
    t1 = time.time()
    res = predictor.batch_predict(imgs)
    print("Total time: ", round(time.time() - t1, 4))
    print(res)