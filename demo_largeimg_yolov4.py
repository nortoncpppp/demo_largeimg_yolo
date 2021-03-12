from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from tqdm import tqdm

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def convertBack_float(x, y, w, h):
    xmin = (x - (w / 2))
    xmax = (x + (w / 2))
    ymin = (y - (h / 2))
    ymax = (y + (h / 2))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        # x, y, w, h = detection[2][0],\
        #     detection[2][1],\
        #     detection[2][2],\
        #     detection[2][3]
        # xmin, ymin, xmax, ymax = convertBack(
        #     float(x), float(y), float(w), float(h))
        xmin, ymin, xmax, ymax = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def cvDrawBoxes_yolo(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    length = len(dets)
    all_scores = np.zeros(length)
    all_areas = np.zeros(length)
    all_x1 = np.zeros(length)
    all_x2 = np.zeros(length)
    all_y1 = np.zeros(length)
    all_y2 = np.zeros(length)
    for index,det in enumerate(dets):
        x1 = det[2][0]
        y1 = det[2][1]
        x2 = det[2][2]
        y2 = det[2][3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        scores = det[1]

        all_scores[index] = scores
        all_areas[index] = areas
        all_x1[index] = x1
        all_x2[index] = x2
        all_y1[index] = y1
        all_y2[index] = y2

        ## index for dets
    order = all_scores.argsort()

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(all_x1[i], all_x1[order[1:]])
        yy1 = np.maximum(all_y1[i], all_y1[order[1:]])
        xx2 = np.minimum(all_x2[i], all_x2[order[1:]])
        yy2 = np.minimum(all_y2[i], all_y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (all_areas[i] + all_areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

netMain = None
metaMain = None
altNames = None

class YOLO():
    def __init__(self, config_file, weight_file, metafile):
        global metaMain, netMain, altNames
        self.configPath = config_file
        self.weightPath = weight_file
        self.metaPath = metafile
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.metaPath)+"`")
        if netMain is None:
            netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(netMain),
                                        darknet.network_height(netMain),3)

    def inference_single(self, imagname, slide_size, chip_size):
        img = cv2.imread(imagname)
        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = []

        for i in tqdm(range(int(width / slide_w ))):
            for j in range(int(height / slide_h)):
                # subimg = np.zeros((hn, wn, channel))
                # print('i: ', i, 'j: ', j)
                chip_h_max = j*slide_h + hn
                chip_w_max = i*slide_w + wn
                if chip_h_max >= height:
                    chip_h_max = height
                if chip_w_max >= width:
                    chip_w_max = width
                chip = img[j*slide_h:chip_h_max, i*slide_w:chip_w_max, :3]
                # subimg[:chip.shape[0], :chip.shape[1], :] = chip
                # chip_detections = inference_detector(self.model, subimg)
                frame_rgb = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb,
                                           (darknet.network_width(netMain),
                                            darknet.network_height(netMain)),
                                           interpolation=cv2.INTER_LINEAR)
                slice_h,slice_w,slice_c = frame_rgb.shape
                scale_ratio_w = slice_w / darknet.network_width(netMain)
                scale_ratio_h = slice_h / darknet.network_height(netMain)

                darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

                chip_detections = darknet.detect_image(netMain, metaMain, self.darknet_image, thresh=0.25)
                image = cvDrawBoxes_yolo(chip_detections, frame_resized)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imshow('Demo', image)
                cv2.waitKey()

                for detection in chip_detections:
                    x, y, w, h = detection[2][0], \
                                 detection[2][1], \
                                 detection[2][2], \
                                 detection[2][3]
                    xmin, ymin, xmax, ymax = convertBack_float(
                        float(x), float(y), float(w), float(h))
                    #xmin ymin xmax ymax
                    xmin_large = xmin*scale_ratio_w + i * slide_w
                    ymin_large = ymin*scale_ratio_h + j * slide_h
                    xmax_large = xmax*scale_ratio_w + i * slide_w
                    ymax_large = ymax*scale_ratio_h + j * slide_h

                    total_detections.append((detection[0],detection[1],(xmin_large,ymin_large,xmax_large,ymax_large)))

        # image = cvDrawBoxes(total_detections, img)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Demo', image)
        # cv2.waitKey()
        # nms
        keeps = py_cpu_nms(total_detections, 0.1)
        total_detections_keep = []
        for keep in keeps:
            total_detections_keep.append(total_detections[keep])

        image = cvDrawBoxes(total_detections_keep, img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Demo', image)
        cv2.waitKey()
        cv2.imwrite('result.jpg', image)

        return total_detections_keep


if __name__ == '__main__':
    prev_time = time.time()
    yolo_model = YOLO('./yolov4-mstar-car.cfg','./yolov4-mstar-car_best.weights','./car.data')
    print(1 / (time.time() - prev_time))
    yolo_model.inference_single('demo.jpg',(512, 512),(1024, 1024))
    print(1 / (time.time() - prev_time))
