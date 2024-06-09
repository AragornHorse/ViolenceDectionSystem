import config
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import random

max_width = config.max_width
max_time = config.max_time
dt = 2


def random_clip():
    if random.random() > config.clip_prob:
        return [0, 0], 1.
    else:
        width = 1 - random.random() * 0.4    # 0.6 ~ 1.
        x = max([random.random() * (0.8 - width), 0])    # 0 ~ 0.8 - width
        y = max([random.random() * (1 - width), 0])    # 0 ~ 1 - width
        return [x, y], width


def mp4_to_array(pth, clip=False, clip_start=None, clip_rate=1., resize=True):

    if clip:
        if clip_start is None:
            clip_start = [0, 0]
        width = None

    video = cv.VideoCapture(pth)
    frames = []
    while True:
        for _ in range(dt):
            ret, frame = video.read()
            if not ret or len(frames) >= max_time:
                break
        if frame is None or not ret or len(frames) >= max_time:
            break

        if clip:
            if width is None:
                width = int(clip_rate * max(frame.shape))
                clip_start[0] = int(clip_start[0] * max(frame.shape))
                clip_start[1] = int(clip_start[1] * max(frame.shape))
                if clip_start[0] + width > frame.shape[0]:
                    clip_start[0] = max([0, frame.shape[0] - width])
                if clip_start[1] + width > frame.shape[1]:
                    clip_start[1] = max([0, frame.shape[1] - width])
            frame = frame[clip_start[0]: clip_start[0] + width, clip_start[1]: clip_start[1] + width]

        if max(frame.shape) > max_width and resize:
            rate = max_width / max(frame.shape)
            frame = cv.resize(frame, (int(frame.shape[1] * rate), int(frame.shape[0] * rate)))
        frame = np.array(frame)
        frames.append(frame)
    video.release()
    frames = np.array(frames).transpose([3, 0, 1, 2])[[2, 1, 0], ...]
    return frames


def mp4_to_tensor(pth, device=torch.device("cuda"), clip=False, clip_start=None, clip_rate=1.):
    array = mp4_to_array(pth, clip=clip, clip_start=clip_start, clip_rate=clip_rate)
    return torch.tensor(array, dtype=torch.float, device=device)


def jpgs_to_array(pth):
    path = glob.glob(pth + '/*.jpg')
    lst = sorted(path, key=os.path.getctime)
    imgs = [cv.imread(pth) for pth in lst]

    if max(imgs[0].shape) > max_width:
        rate = max_width / max(imgs[0].shape)
        imgs = [cv.resize(im, (int(im.shape[1] * rate), int(im.shape[0] * rate))) for im in imgs]
    imgs = np.stack(imgs).transpose([3, 0, 1, 2])
    return imgs


def jpgs_to_tensor(pth, device=torch.device("cuda")):
    array = jpgs_to_array(pth)
    return torch.tensor(array, dtype=torch.float, device=device)


def mp4_to_jpgs(mp4_pth):
    from PIL import Image
    video = mp4_to_array(mp4_pth).transpose([1, 2, 3, 0])

    for i, img in enumerate(video):
        image = Image.fromarray(img)
        image.save('./jpgs/image{}.jpg'.format(i))
        image = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite('./jpgs/image{}.jpg'.format(i), image)


if __name__ == '__main__':

    # pth = r"C:\Users\DELL\Desktop\datasets\Real Life Violence Situations Dataset\archive\Real Life Violence Dataset\Violence\V_1.mp4"
    #
    # video = mp4_to_array(pth, clip=True, clip_start=[0.5, 0.5], clip_rate=0.4)
    # print(video.shape)
    # plt.imshow(np.transpose(video, [1, 2, 3, 0])[0])
    # plt.show()
    # plt.imshow(np.transpose(video, [1, 2, 3, 0])[3])
    # plt.show()

    for i in range(100):
        print(random_clip())