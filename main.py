import resnet_notail as resnet
import torch
import torch.nn as nn
import i3d
import numpy as np
import pickle
import load_data
import config
import temporal
import get_audio_feature
import moving_agent


def bayes_threshold(w=1.02):
    th = (1 - config.p_fight) / config.p_fight * (config.C01 - config.C00) / (config.C10 - config.C11)
    p_th = th / (1 + th)
    return np.log((p_th + 1e-30) / (1 - p_th + 1e-30)) / (w + 1e-30), p_th


# cpu or gpu
device = torch.device(config.device if torch.cuda.is_available() else 'cpu')


# load video model
if config.use_video:
    # load classifier
    svm = None
    pth = None
    if config.backbone[0] == 'I':
        pth = r"./para/{}_i{}t{}p.pkl".format(config.model, config.dt, config.max_width)
    elif config.backbone[0] == 'R':
        pth = r"./para/{}_r{}{}t{}p.pkl".format(config.model, config.backbone[-2:], config.dt, config.max_width)
    else:
        import warnings
        warnings.warn("illegal model: {}".format(config.model))

    with open(pth, 'rb') as f:
        svm = pickle.load(f)

    # load CNN3D
    if config.backbone[0] == 'I':
        model = i3d.generate_model(device)
    else:
        model = resnet.generate_model(
            int(config.backbone[-2:]), device, pertain=True, n_classes=700, n_input_channels=3
        )

    model.eval()


# load audio model
if config.use_audio:

    audio_model = nn.Sequential(
        nn.Dropout(p=0.),
        nn.BatchNorm2d(1),
        nn.Conv2d(1, 2, 3, 2, 0),  # 200
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(2, 3, 3, 2, 0),  # 100
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(3),
        nn.Conv2d(3, 4, 3, 2),  # 50
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(4, 6, 3, 2),  # 24
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(6, 8, 3, 2),  # 24
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(8, 10, 3, 2),  # 24
        nn.Flatten(),
        nn.Linear(50, 1),
        nn.Sigmoid()
    ).to(device)

    audio_model.load_state_dict(torch.load(r"./para/audio_400_128.pth"))
    audio_model.eval()


# moving agent
agent = None
if config.agent == "bayes":
    agent = moving_agent.BayesAgent()
if config.agent == 'policy-gradient':
    agent = moving_agent.KernelPolicyGradientAgent()
if config.agent == 'grad':
    agent = moving_agent.GradAgent()


# generate bayes threshold
threshold, p_threshold = bayes_threshold()
# print(threshold, p_threshold)


# generate temporal filter
temp_filter = None
if config.temporal_filter == 'auto-machine-3':
    temp_filter = temporal.AutoMachine()
if config.temporal_filter == 'hmm':
    temp_filter = temporal.HMM()

# cache
# temporary
t = 0
last_p_video = 0.
last_clip_start = [0, 0]
last_clip_width = 1.


def recognize():

    global t, last_clip_width, last_p_video, last_clip_start

    pth = r"C:\Users\DELL\Desktop\datasets\Real Life Violence Situations Dataset\archive\Real Life Violence Dataset\Violence\V_10.mp4"

    # initial prediction is 0.5
    pred = 0.5
    p_video = 0.5

    # calculate p_video
    if config.use_video:

        # calculate from the entire area of images
        if not config.clip:
            imgs = load_data.mp4_to_tensor(pth, device=device)[None, ...] / 255 * 2 - 1
        # clip a part of image to locally enlarge
        else:
            # if last clip is violent, clip the same area
            if last_p_video > 0.5:
                imgs = load_data.mp4_to_tensor(
                    pth, device=device, clip=True, clip_start=last_clip_start, clip_rate=last_clip_start
                )[None, ...] / 255 * 2 - 1
            # clip randomly
            else:
                start, w = load_data.random_clip()
                imgs = load_data.mp4_to_tensor(
                    pth, device=device, clip=True, clip_start=start, clip_rate=w
                )[None, ...] / 255 * 2 - 1
                # save the clip position
                last_clip_start = start
                last_clip_width = w

        # extract video features by 3d-cnn
        with torch.no_grad():
            feature = model(imgs).detach().cpu().numpy()    # 1, h

        # calculate p_video by svm and sigmoid
        p_video = (1 / (np.exp(-1.02 * float(svm.decision_function(feature))) + 1))
        # save p_video in this clip
        last_p_video = p_video

        # only video
        if not config.use_audio:
            pred = p_video
        # audio-video
        else:
            pred = p_video ** config.video_w

    # print(p_video)

    # calculate p_audio
    if config.use_audio:

        # get audio from mp4
        audio = get_audio_feature.mp4_to_freq_array(pth)

        # no audio in mp4
        if audio is None:
            print("no audio in {}".format(pth))
            pred = p_video
        # have audio
        else:
            # audio not load enough (0.03 db)
            if isinstance(audio, int) and audio == -1:
                print("audio is not loud enough in {}".format(pth))
                pred = p_video
            # legal audio, exist and loud enough
            else:
                audio = torch.tensor(audio, dtype=torch.float, device=device)[None, None, ...]

                # calculate p_audio by 2d-cnn
                # p_audio isn't reliable enough
                p_audio = float(np.clip(audio_model(audio).cpu().detach().numpy(), a_min=0.35, a_max=0.7))

                # only audio
                if not config.use_video:
                    pred = p_audio
                # audio-video
                else:
                    pred = (pred * p_audio ** config.audio_w) ** (1 / (config.audio_w + config.video_w))

    # print(p_audio)
    # print(pred)

    # initial moving is to stay
    a = [0, 0]

    # calculate next step
    if agent is not None and last_p_video < 0.5:
        # not moving turn
        if t < config.step_per_move:
            t += 1
        # calculate next step
        else:
            # needing video to calculate
            if isinstance(agent, moving_agent.KernelPolicyGradientAgent) and config.visual_agent:
                a = agent.next_step(pred, now_video=feature)
            # just need last position to calculate
            else:
                a = agent.next_step(pred)

    # eliminate mis-identification
    if temp_filter is not None:
        if config.temporal_filter == 'hmm':
            pred = temp_filter.update(pred, has_sigmoid=True)
        else:
            pred = temp_filter.update(int(pred > p_threshold))

    print(pred, a)


recognize()
