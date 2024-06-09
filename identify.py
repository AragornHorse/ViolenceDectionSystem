from ultralytics.models.yolo import model as M
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


# a block of reid model
# you don't need to read this class
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * stride, 3, stride, 1, bias=False),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channel * stride, out_channel, 1, 1, 0, bias=False),
            nn.LeakyReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )

        self.shortcut = nn.Identity() \
            if stride == 1 and in_channel == out_channel else \
            nn.Conv2d(in_channel, out_channel, 3, stride, 1)

        self.norm = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        out = self.norm(out)
        return out


# load yolo to detect persons
# download the parameters automatically
model = M.YOLO("./para/yolov3.pt")

# the reid model to identify persons' attributes, like gender, wearings
# don't modify its structure
# input an image of one person, output its attributes
# out = [attr_1, attr_2, attr_3, ...]
# attr_i is the probability that person satisfies this attribute
reid_model = nn.Sequential(
    nn.Conv2d(3, 4, 3, 1, 1),
    nn.BatchNorm2d(4),
    nn.LeakyReLU(True),
    ResBlock(4, 6, 2),
    ResBlock(6, 8, 2),
    ResBlock(8, 10, 2),
    ResBlock(10, 12, 2),
    ResBlock(12, 14, 2),
    nn.AdaptiveMaxPool2d(1),
    nn.Flatten(),
    nn.Linear(14, 26),
    nn.Sigmoid()
)
# load parameters trained from PA-100k
reid_model.load_state_dict(torch.load(r"./para/reid.pth"))

# max frame number of one video
max_time = 64
# max width of one image
max_width = 512
# time interval between two frames
dt = 2


# don't need to modify this function,
# but you should know its input and output
def get_person_position(pth, plot=False, return_video=True):
    """
        input a video, output each person's location in each frame

    :param pth:             mp4 path
    :param plot:            show each frame by pyplot or not
    :param return_video:    return the video or don't return the video

    :return:
        1. rst: a 3-dim list

            rst[i][j] = [x1, y1, x2, y2] is the detected result for the j-th person in the i-th frame

            x1, y1, x2, y2:
                    (x1, y1)______
                       |    O    |     ---> x
                       |   /|)   |       ^  y
                       |    |    |
                       |___/_L__(x2, y2)

        2. frames:  array shaped (channel, frame_number, height, width)
    """
    video = cv.VideoCapture(pth)
    k = 0
    rsts = []

    frames = []

    while True:
        k += 1
        if k % dt != 0:
            continue
        ret, frame = video.read()
        if not ret or k >= max_time:
            break
        if max(frame.shape) > max_width:
            rate = max_width / max(frame.shape)
            frame = cv.resize(frame, (int(frame.shape[1] * rate), int(frame.shape[0] * rate)))
        rst = model.predict(frame)[0]
        names = rst.names
        rsts.append([])

        if return_video:
            frames.append(np.array(frame))

        if plot:
            plt.imshow(frame)

        for item in rst:
            x1, y1, x2, y2 = item.boxes.xyxy.cpu().numpy()[0]
            cls = int(item.boxes.cls)
            conf = float(item.boxes.conf)
            if names[cls] == 'person':
                if plot:
                    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=1.)
                    # plt.text(x1, y1, "{:.2f}, {}".format(conf, names[cls]))
                # conf = float(item.boxes.conf)
                # plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=1.)
                rsts[-1].append([x1, y1, x2, y2])
        if plot:
            plt.show()

    video.release()

    if return_video:
        frames = np.array(frames).transpose([3, 0, 1, 2])[[2, 1, 0], ...]
        return rsts, frames
    else:
        return rsts


# you should finish this function
def get_person(pth):
    """
        input a video, output every persons' picture and each person's attributes

            [
                [
                    ________
                    |   O  |
                    |  /|) |
                    |  /L  |  ,    [attribute1, attribute2, ...]
                    |______|
                ],
                ... (other persons)
            ]

    :param pth:   mp4 path
    :return: a list
        rst: rst[i] = [img, attrs] is the i-th person's image and his attributes
            img is a np-array or other format
            attrs = [attr1, attr2, ...]
    """

    # video.shape == (channel, frame_number, height, width)
    # location[i][j] == [x1, y1, x2, y2], is the position of the j-th person in the i-th image
    # more details for video and location is in def get_person_position()
    locations, video = get_person_position(pth, plot=False, return_video=True)

    # you should define i: which image in this video will be used to identify persons
    image = video[:, i, ...]

    # you should clip out all the persons
    # resize these images to (96, 32)
    # use 0 to pad
    # you can draw lessons from reid.load.PA100k.getitem()
    # but in reid.load.PA100k.getitem(), an image is an opencv array, and now image is a np-array
    # maybe you can transform image to opencv first
    persons = ["xxx"]   # l list of images

    # turn to torch.tensor
    # its shape must be (person_number, channel, height, width)
    # you can use torch.transpose to change the shape
    person = torch.tensor(persons, dtype=torch.float)

    # the probability each person satisfies each attribute
    # idents.shape: (person_number, attribute_number)
    # attributes: [
    #   0: female, 1: age-over-60, 2: age-18-60, 3: age<18, 4: front, 5: side, 6: back, 7: hat, 8: glasses,
    #   9: hand-bag, 10: shoulder-bag, 11: backpack, 12: holding-objects, 13: short-sleeve, 14: long-sleeve,
    #   15: upper-stride, 16: upper-logo, 17: upper-plaid, 18: upper-splice, 19: lower-stripe, 20: lower-pattern
    #   21: long-coat, 22: trousers, 23: shorts, 24: skirt&dress, 25: boots
    # ]
    # trained from PA-100k dataset, you can check that dataset for more details
    idents = reid_model(person)

    # format the result
    rst = [
        [persons[i], attrs]
        for i in range(len(persons))
    ]

    return rst


if __name__ == '__main__':
    pth = r"C:\Users\DELL\Desktop\datasets\UBI_FIGHTS\videos\fight\F_41_0_0_0_0.mp4"

    print(get_person_position(pth, plot=True)[1].shape)


