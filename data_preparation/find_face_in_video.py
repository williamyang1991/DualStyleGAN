import dlib
from PIL import Image
import time
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import multiprocessing as mp
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

savename = '1'
videopath = './video/Arcane1.mp4'
outpath = './output/'
predictorpath = './shape_predictor_68_face_landmarks.dat'
black_width = 89

def align_face(img, lm):
    """
    :param filepath: str
    :return: PIL Image
    """

    #lm = get_landmark(filepath, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    #img = PIL.Image.open(filepath)

    output_size = 512
    transform_size = 512
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img

def find_face(videoname, savename, detector, predictor):
    videoCapture = cv2.VideoCapture(videoname)
    success = True
    i = 0
    j = 0
    while success:
        i = i + 1
        success, frame = videoCapture.read()
        if i % 24 > 0: # try to find a face every 24 frames
            continue
        # crop the black regions in the movie
        if black_width > 0:
            frame = cv2.cvtColor(frame[black_width:-black_width], cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        dets = detector(frame, 1)
        findlm = False
        for k, d in enumerate(dets):
            if d.width() >= 256 and d.height() >= 256:
                shape = predictor(frame, d)
                findlm = True
                break
        if findlm == False:
            continue

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        img = Image.fromarray(frame)
        result = align_face(img, lm)
        j = j + 1
        result.save(os.path.join(outpath, "%s_%03d_%05d.jpg"%(savename, j, i)), quality=95)
    videoCapture.release()    


if __name__ == "__main__":
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictorpath)

    find_face(videopath, savename, detector, predictor) 
