import onnxruntime
from PIL import Image
import numpy as np
import cv2
from scipy.special import softmax
import numpy as np 
import torch
import torchvision
import time
import os

IMAGE_SIZE = (640,640)
RATE = 0.85

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.25, classes=None, agnostic=True, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect_batch_frame(session, images=None, image_size=(512,512), mpt_img=None):
    img_ins = []
    for image_src in images:
        # resized = cv2.resize(image_src, image_size)
        resized = letterbox(image_src, new_shape=image_size)[0]
        img_in = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        img_in /= 255.0
        img_ins.append(img_in)
    
    input_name = session.get_inputs()[0].name
    img_ins = np.array(img_ins)
    outputs = session.run(None, {input_name: img_ins})
    # print('output: ', outputs)

    nms_outputs = []
    for img_output in outputs[0]:
        batch_detections = torch.from_numpy(np.array([img_output]))
        batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.45, agnostic=False)
        nms_outputs.append(batch_detections)
    # print(len(nms_outputs))
    # print('nms_outputs: ', nms_outputs)
    
    nms_outputs[0][0][:, :4] = scale_coords(img_ins.shape[2:], nms_outputs[0][0][:, :4], image_src.shape).round()
    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
    if mpt_img is not None:
        boxes = nms_outputs[0][0][:,:4]
        pred = nms_outputs[0][0][:,-1]
        if len(pred):
            pred = 1
        else: 
            pred = 0
        for box in boxes:
            box[0] = 0
            box[2] = image_src.shape[1]
            crop = save_one_box(box, image_src, save=False)
            im_pil = Image.fromarray(crop)
            mpt_img.paste(im_pil,(int(box[0]),int(box[1])))
        return pred, mpt_img

    else:
        pos_pred_count = 0
        labels = []
        # boxs =[]
        # print('nms_outputs: ', nms_outputs)
        for nms_output in nms_outputs:
            # print(nms_output)
            if nms_output[0] is not None and len(nms_output[0]) > 0:
                labels = nms_output[0][..., -1]
                # boxs = nms_output[0][..., :4]
                # pos_pred_count += int(max(labels).item())

        return labels
        
def block_split(lst, block_size, sample):
    stride = (block_size // sample) + 1
    for i in range(0, len(lst), block_size):
        if len(lst[i:i + block_size]) < sample:
            yield lst[i:i + block_size]
        else:
            yield lst[i:i + block_size:stride][:sample]

def laydown(model, list_frames = []):
    number_frame_laydown = detect_batch_frame(model, list_frames, image_size=IMAGE_SIZE)
    return number_frame_laydown / len(list_frames) >= RATE

def check_laydown(model, opt, frame_subfolder_path):
    # count number seconds laydown
    count = 0    

    # load config
    sample = opt.sample
    vid_fps = opt.fps
    source = frame_subfolder_path
    threshold = opt.threshold
    block_size = opt.block_size

    # time of one block
    block_time = block_size / vid_fps

    # list all of frames
    frames = [os.path.join(source, img) for img in sorted(os.listdir(source)) if img.endswith(('.png', '.jpg', '.PNG', '.JPG'))]
    
    for block in block_split(frames, block_size, sample):
        #if have human then check laydown otherwise reset laydown
        if laydown(model, list_frames = block):            
            count += block_time
        else:
            count = 0
            
        if count >= threshold:
            print("Laydown")
        else: 
            print("Normal")

def overlay_bbox_cv(img, all_box, class_names):
    """Draw result boxes
    Copy from nanodet/util/visualization.py
    """
    # all_box array of [label, x0, y0, x1, y1, score]
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[label], score * 100)
        txt_color=(0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(img,
                      (x0, y0 - txt_size[1] - 1),
                      (x0 + txt_size[0] + txt_size[1], y0 - 1), color, -1)
        cv2.putText(img, text, (x0, y0-1),
                    font, 0.5, txt_color, thickness=1)
    return img
def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite('crop.jpg', crop)
    return crop
# if __name__ == '__main__':
#     img = cv2.imread('/content/test_8.png')
#     provider = os.getenv('PROVIDER', 'CUDAExecutionProvider')
#     # provider = os.getenv('PROVIDER', 'CPUExecutionProvider')
#     model = onnxruntime.InferenceSession("models/yolov5/best.onnx", providers=[provider])
#     label = detect_batch_frame(model, [img], image_size=(1280,1280))
#     print(label)


