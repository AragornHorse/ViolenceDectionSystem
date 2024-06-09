import numpy as np


class AutoMachine:
    def __init__(self):
        self.state = '000'

    def update(self, is_fight):
        """
        :param is_fight:   0/1
        :return:    0: non-fight, 1: attention, 2: might fight, 3: fighting, -1: error
        """
        is_fight = int(is_fight)
        self.state = self.state[1:] + str(is_fight)
        if self.state in ['111']:
            return 3
        elif self.state in ['110', '101', '011']:
            return 2
        elif self.state in ['001']:
            return 1
        elif self.state in ['000', '010', '100']:
            return 0
        else:
            return -1


class HMM:
    def __init__(self, p0=0.5, p00=0.7, p11=0.6):
        """
        :param p0:     initial prob of fighting
        :param p00:    p(t=non-fight| t-1=non-fight)
        :param p11:    p(t=fight| t-1=fight)
        """
        self.trans_prob = np.array([
            [p00, 1-p00],
            [1-p11, p11]
        ])

        self.alpha = np.array([1-p0, p0])[:, None]

    def update(self, fight_prob, has_sigmoid=True, w=1.02):
        if not has_sigmoid:
            fight_prob = 1. / (np.exp(-fight_prob * w) + 1.)

        p_y = np.array([1-fight_prob, fight_prob])[:, None]

        # print(self.alpha, '\n', self.trans_prob, '\n', p_y)
        self.alpha = self.trans_prob.T @ self.alpha * p_y
        self.alpha = self.alpha / np.sum(self.alpha)

        p = self.alpha[1, 0]

        if p < 0.5:
            return 0
        elif p < 0.6:
            return 1
        elif p < 0.75:
            return 2
        else:
            return 3


if __name__ == '__main__':
    lst = [0.7, 0.4, 0.4, 0.6, 0.5, 0.4, 0.7, 0.9, 0.3, 0.5]
    model = HMM()
    for i in lst:
        print("{}, {}".format(i, model.update(i, has_sigmoid=True)))



