import numpy as np
import scipy.ndimage
import os
import PIL.Image

txtpath = './caricature.txt'
lmpath = './WebCaricature/FacialPoints/'
impath = './WebCaricature/OriginalImages/'
outpath = './Caricature/'

def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        '''
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise
        '''
        lm_eye_left = lm[8:10]
        lm_eye_right = lm[10:12]
        lm_mouth_outer = lm[13:17]
        
        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        '''
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        '''
        mouth_left   = lm_mouth_outer[1]
        mouth_right  = lm_mouth_outer[3]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = np.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file).convert('RGBA').convert('RGB')

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        #border = max(int(np.rint(qsize * 0.7)), 1)
        border = max(int(np.rint(qsize * 0.01)), 0.5)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 2:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            #pad = np.maximum(pad, int(np.rint(qsize * 0.05)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'edge')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            #initial was *0.2
            blur = qsize * 0.002
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))
            if alpha:
                mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
                mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
                img = np.concatenate((img, mask), axis=2)
                img = PIL.Image.fromarray(img, 'RGBA')
            else:
                img = PIL.Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file, 'PNG')


if __name__ == "__main__":
	if not os.path.exists(outpath):
	    os.makedirs(outpath)

	f = open(txtpath,'r')
	names = f.readlines()
	f.close()

	for name in names:
	    fname = name[:-8]
	    num = name[-7:-1]
	    lm = np.loadtxt(os.path.join(lmpath, fname.replace('_', ' '), num + '.txt'))
	    imgname = os.path.join(impath, fname.replace('_', ' '), num + '.jpg')
	    savename = os.path.join(outpath, fname + '_' + num + '.jpg')
	    image_align(imgname, savename, lm, output_size=256, transform_size=1024)