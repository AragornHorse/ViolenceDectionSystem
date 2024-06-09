
# the max width or height of one frame
# 128 / 256
max_width = 128

# the max frame number of one video
# 64 / 48 / 32
max_time = 48

# clip a part of the images
clip = True
clip_prob = 0.5

# the time interval between two frame (24/s)
# 2 / 3 / 4
dt = 2

# model for feature extracting
# R3D34 / R3D50 / I3D
backbone = 'R3D34'

# model for recognition
# svm / dnn
model = 'svm'

# prior probability for fighting
p_fight = 0.5

# costs
C01 = 1.  # mistake non-fight to fight
C10 = 1.  # mistake fight to non-fight
C00 = 0.  # correctly recognized fight
C11 = 0.  # correctly recognized non-fight

# temporal filter
# auto-machine-3 / none / hmm
temporal_filter = 'auto-machine-3'

# device for CNNs
device = 'cpu'

# use audio, video or not
use_audio = True
use_video = True

audio_w = 1.
video_w = 5.

# agent algorithm to move camera, bayes, policy-gradient, grad or none
agent = 'bayes'
# agent can see the video
visual_agent = False
# how many steps between two move
step_per_move = 5
