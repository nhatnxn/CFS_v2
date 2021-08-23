import time
from PIL import Image
import cv2
import numpy as np
import math
import torch
from collections import defaultdict
import os

from .translate import build_model, translate, process_input
from .utils import download_weights

class Predictor():
    def __init__(self, config):

        device = config['device']

        model, vocab = build_model(config)
        
        if os.path.exists(config['weights_local']):
            weights = config['weights_local']
        else:
            if config['weights'].startswith('http'):
                weights = download_weights(config['weights'])
            else:
                weights = config['weights']


        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img):
        '''

        :param img:
        :return:
        '''
        img = process_input(img, self.config['dataset']['image_height'],
                            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'],
                            is_padding=self.config['dataset']['is_padding'], round_to=self.config['dataset']['round_to'])
        img = img.to(self.config['device'])

        s, prob = translate(img, self.model, get_prob=False)
        s = s[0].tolist()
        s = self.vocab.decode(s)
        return s

    def batch_predict(self, images):
        """
        Recognize images on batch

        Parameters:
        images(list): list of cropped images
        set_buck_thresh(int): threshold to merge bucket in images

        Return:
        result(list string): ocr results
        """
        batch_dict, indices = self.batch_process(images)

        list_keys = [i for i in batch_dict if batch_dict[i]
                     != batch_dict.default_factory()]
        result = list([])

        for width in list_keys:
            batch = batch_dict[width]
            batch = np.asarray(batch)
            batch = torch.FloatTensor(batch)
            batch = batch.to(self.config['device'])
            sent = translate(batch, self.model)[0].tolist()
            batch_text = self.vocab.batch_decode(sent)
            result.extend(batch_text)

        # sort text result corresponding to original coordinate
        z = zip(result, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        result, _ = zip(*sorted_result)

        return result

    def preprocess_input(self, img, image_height, image_min_width, image_max_width, round_to=True,
                  padding_type='right'):
        """
        Preprocess input image (resize, normalize)

        Parameters:
        image: has shape of (H, W, C)   :cv2 Image

        Return:
        img: has shape (H, W, C)
        """

        # h, w, _ = image.shape
        # new_w, image_height = self.resize_v1(w, h, self.config['dataset']['image_height'],
        #                                      self.config['dataset']['image_min_width'],
        #                                      self.config['dataset']['image_max_width'])
        #
        # img = cv2.resize(image, (new_w, image_height))
        # img = img / 255.0
        # img = np.transpose(img, (2, 0, 1))
        # t1 = time.time()
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # print("Time PIL to cv: ", round(time.time() - t1, 6))
        theta = 0.0001
        w, h = img.size
        new_w = int(image_height * float(w) / float(h))

        if new_w == 0:
            new_w=1

        if new_w < image_max_width:
            # from IPython import embed; embed()
            if new_w != w:
                img = img.resize((new_w, image_height), Image.ANTIALIAS)
            if padding_type == 'center':
                box_paste = ((image_max_width - new_w) // 2, 0)
            elif padding_type == 'right':
                box_paste = (0, 0)
            else:
                raise Exception("Not implement padding_type")

            new_w, image_height = self.get_width_for_cluster(w, h, image_height, image_min_width, image_max_width, round_to=round_to)
            new_img = Image.new('RGB', (new_w, image_height), 'white')  # padding white
            new_img.paste(img, box=box_paste)

            img = new_img

        else:
            # resize
            img = img.resize((image_max_width, image_height), Image.ANTIALIAS)

        img = np.asarray(img).transpose(2, 0, 1)
        img = img / 255

        return img

    def batch_process(self, images, set_bucket_thresh=0):
        """
        Preprocess list input images and divide list input images to sub bucket which has same length

        Parameters:
        image: has shape of (B, H, W, C)
            set_buck_thresh(int): threshold to merge bucket in images

        Return:
        batch_img_dict: list
            list of batch imgs
        indices:
            position of each img in "images" argument
        """

        batch_img_dict = defaultdict(list)
        image_height = self.config['dataset']['image_height']
        image_min_width = self.config['dataset']['image_min_width']
        image_max_width = self.config['dataset']['image_max_width']
        padding_type = self.config['dataset']['padding_type']
        round_to = self.config['dataset']['round_to']
        img_li = [self.preprocess_input(img,  image_height, image_min_width, image_max_width, round_to=round_to,
                  padding_type=padding_type) for img in images]
        img_li, width_list, indices = self.sort_width(img_li, reverse=False)

        for i, image in enumerate(img_li):
            c, h, w = image.shape
            batch_img_dict[w].append(image)

        return batch_img_dict, indices

    @staticmethod
    def sort_width(batch_img: list, reverse: bool = False):
        """
        Sort list image correspondint to width of each image

        Parameters
        ----------
        batch_img: list
            list input image

        Return
        ------
        sorted_batch_img: list
            sorted input images
        width_img_list: list
            list of width images
        indices: list
            sorted position of each image in original batch images
        """

        def get_img_width(element):
            img = element[0]
            c, h, w = img.shape
            return w

        batch = list(zip(batch_img, range(len(batch_img))))
        sorted_batch = sorted(batch, key=get_img_width, reverse=reverse)
        sorted_batch_img, indices = list(zip(*sorted_batch))
        width_img_list = list(map(get_img_width, batch))

        return sorted_batch_img, width_img_list, indices

    @staticmethod
    def get_width_for_cluster(w: int, h: int, expected_height: int, image_min_width: int, image_max_width: int, round_to=50):
        """
        Get expected height and width of image

        Parameters
        ----------
        w: int
            width of image
        h: int
            height
        expected_height: int
        image_min_width: int
        image_max_width: int
            max_width of

        Return
        ------

        """
        new_w = int(expected_height * float(w) / float(h))
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height
