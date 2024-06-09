import numpy as np


def is_positive_definite(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return True
    else:
        return False


class BayesAgent:
    def __init__(self,
                 p_stay=0.5,
                 main_position=np.array([0., 0.]), cov_main_position=np.eye(2)*0.1,
                 init_position=np.array([0., 0.]), cov_stay=np.eye(2)*0.1,
                 w_prefer_more_violence=200,
                 memory=10,
                 step_len=0.3, max_step=1.,
                 lbd=1e-5, kalman=True, gamma=0.1
                 ):
        """
        :param p_stay:                   prob to stay current position
        :param main_position:            the best position you think
        :param cov_main_position:        the cov to forward best position
        :param init_position:            initial position
        :param cov_stay:                 cov to stay current position
        :param w_prefer_more_violence:   the degree for seeking more violence
        :param memory:                   history position number
        :param step_len:                 how long in agent coordinate equals to one step of camera
        :param lbd:                      prior prob for w
        """

        self.p_stay = p_stay
        self.main_pos = main_position[:, None]
        self.cov_main_pos = cov_main_position
        self.cov_stay = cov_stay
        self.w_more_vio = w_prefer_more_violence
        self.step_len = step_len
        self.lbd = lbd

        # x
        self.pos = init_position[:, None]

        # history position
        self.X = np.random.randn(memory, 2) * 1e-2
        # history violence
        self.Y = np.random.random([memory, 1]) * 0.2 + 0.2
        # stack pointer
        self.i = 0

        self.cov_main_pos_inv = np.linalg.inv(cov_main_position)
        self.cov_stay_inv = np.linalg.inv(cov_stay)
        self.max_step = max_step

        # w'x is the prediction for violence
        self.w = None
        self.H = None

        self.kalman = kalman
        self.gamma = gamma

    def next_step(self, now_violence):

        # current (position, violence) into stack
        self.X[self.i, :] = self.pos.reshape([-1])
        self.Y[self.i, 0] = float(now_violence)
        self.i = (self.i + 1) % self.X.shape[0]

        # don't move
        if float(np.random.random([1])) < self.p_stay:
            return [0, 0]
        else:
            # estimate w, w'x = violence
            if not self.kalman:
                w = np.linalg.inv(self.X.T @ self.X + self.lbd * np.eye(self.X.shape[1])) @ self.X.T @ self.Y
                self.w = w
            else:
                if self.w is None:
                    self.H = self.lbd * np.eye(2)
                    self.w = np.zeros([2])[:, None]
                self.H = self.gamma * self.H + self.X.T @ self.X
                w = self.w + np.linalg.inv(self.H) @ self.X.T @ (self.Y - self.X @ self.w)
                self.w = w

            # prob for next position
            cov_inv = self.cov_main_pos_inv + self.cov_stay_inv
            cov = np.linalg.inv(cov_inv)
            miu = cov @ (self.cov_main_pos_inv @ self.main_pos + self.cov_stay_inv @ self.pos + self.w_more_vio * w)

            # positive cov, is gaussian distribution
            next_pos = np.random.multivariate_normal(miu[:, 0], cov, [1]).reshape([-1, 1])
            d_pos = (next_pos - self.pos)

            # max step is 8
            d_pos = np.sign(d_pos) * np.clip(np.abs(d_pos) // self.step_len, -8, 8)  # 2, 1
            self.pos += d_pos

            return d_pos.reshape([-1]).astype(int).tolist()


class GaussianKernel:
    def __init__(self, alpha, bias=0., unitization=False):
        self.alpha = alpha
        self.bias = bias
        self.unitization = unitization

    def __call__(self, x1, x2=None):
        """
            exp[-alpha * ||x1-x2||^2]
        :param x1:  (n1, h)
        :param x2:  (n2, h)
        :return:    (n1, n2)
        """
        if x2 is None:
            x2 = np.copy(x1)

        assert x1.shape[1] == x2.shape[1]

        n1, h = x1.shape
        n2, h = x2.shape

        dis = x1.reshape([n1, 1, h]) - x2.reshape([1, n2, h])
        dis = np.sum(dis ** 2, axis=2).reshape([n1, n2])
        dis = np.exp(- self.alpha * dis) + self.bias

        if self.unitization:
            dis = dis / np.sqrt(np.sum(dis ** 2, axis=1, keepdims=True))

        return dis


class LinearKernel:
    def __init__(self, c=0.):
        self.c = c

    def __call__(self, x1, x2=None):
        return x1 @ x2.T + self.c


class KernelPolicyGradientAgent:
    def __init__(self,
                 p_stay=0.5, degree_w=1., memory=10, cov_move=np.eye(2),
                 kernel_func=GaussianKernel(alpha=0.1, bias=0.1)
                 ):
        """
        :param p_stay:       prob to stay current position
        :param degree_w:     prior prob for w, the bigger, the more trend to more violence
        :param memory:       history condition number
        :param cov_move:     cov to move
        :param kernel_func:  kernel function
        """
        self.p_stay = p_stay
        self.degree_w = degree_w
        self.memory = memory
        self.cov = cov_move
        self.kernel_func = kernel_func

        self.x = None   # 11 videos
        self.a = np.zeros([memory+1, 2])         # 11 actions
        self.y = np.zeros([memory+1, 1]) + 0.3   # 11 d_violences
        self.i = 0

        self.last_violence = 0.3
        self.pos = np.array([0., 0.])

    def next_step(self, now_violence, now_video=None):

        # default is position
        if now_video is None:
            now_video = self.pos

        now_video = now_video.reshape([-1])
        now_violence = float(now_violence)

        # stay where it was
        if float(np.random.random([1])) < self.p_stay:
            return [0, 0]
        # move
        else:
            # initial stack
            if self.x is None:
                self.x = np.random.randn(self.memory + 1, now_video.shape[0]) * 1e-2

            # video, violence, action into stack
            self.x[self.i, :] = now_video
            self.y[self.i-1, 0] = now_violence - self.last_violence
            self.last_violence = now_violence

            # usable data
            idx = [i for i in range(self.memory + 1) if i != self.i]
            x = self.x[idx]
            y = self.y[idx]
            a = self.a[idx]

            # print(x, a, y)

            # kernel ridge regression
            G = (y.reshape([-1]) - max(np.mean(y), 0.5))
            G[G < 0] = -1.

            # G[G <= 0] = -10000
            miu = (self.kernel_func(now_video[None, :], x) @
                  np.linalg.inv(self.degree_w * np.diag(1 / (G + 1e-300))
                                + self.kernel_func(x, x)) @ a).T

            # action
            d_pos = np.random.multivariate_normal(miu[:, 0], self.cov, [1]).reshape([-1])
            d_pos = np.clip(d_pos, -8, 8)

            # if bad try, goback to (0,0)
            if np.max(y) < 0.:
                d_pos = np.clip(self.pos, -8, 8)

            self.a[self.i, :] = d_pos

            d_pos = d_pos.astype(int).tolist()
            self.pos += d_pos
            self.i = (self.i + 1) % (self.memory + 1)

            return d_pos


class GradAgent:
    def __init__(self, memory=10, cov=np.eye(2), lbd=1e-2, p_stay=0.5, step=10.):
        self.pos = np.array([0., 0.])
        self.memory = memory
        self.cov = cov
        self.x = np.random.randn(memory, 2) * 0.01
        self.y = np.zeros([memory, 1]) + 0.2
        self.p_stay = p_stay
        self.i = 0
        self.lbd = lbd
        self.w = None
        self.step = step

    def next_step(self, now_violence):
        self.x[self.i, :] = self.pos
        self.y[self.i, 0] = now_violence

        self.i = (self.i + 1) % self.memory

        if float(np.random.random([1])) < self.p_stay:
            return [0, 0]

        self.w = np.linalg.inv(self.x.T @ self.x + self.lbd * np.eye(2)) @ self.x.T @ self.y

        d_pos = self.w.reshape([-1])

        d_pos = (np.random.multivariate_normal(d_pos, self.cov, [1]).reshape([-1]))

        d_pos = np.clip(d_pos * self.step, a_min=-8, a_max=8)

        # print(d_pos)

        d_pos = np.sign(d_pos) * np.abs(d_pos).astype(int)

        self.pos += d_pos

        return d_pos


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from math_model.package import plots

    # model = BayesAgent(p_stay=0.)
    #
    # poses = [[0, 0]]
    #
    # for i in range(10):
    #     pos = model.pos.reshape([-1])
    #
    #     v = - pos[0] + pos[1] + 0.1 * float(np.random.randn(1))
    #     a = model.next_step(v)
    #
    #     poses.append(pos.tolist())
    #
    #     print(pos, v)
    #
    # poses = np.array(poses)
    # c = np.linspace(0.2, 0.9, poses.shape[0])
    # for i in range(poses.shape[0]-1):
    #     plt.plot([poses[i, 0], poses[i+1, 0]], [poses[i, 1], poses[i+1, 1]], linewidth=0.5, color=[0., 0.1, 0.4, c[i]])
    # plt.scatter(poses[:, 0], poses[:, 1], s=3., c=[[0.1, 0.1, 0.6, i] for i in np.linspace(0.1, 0.9, poses.shape[0])])
    # plt.show()

    model = KernelPolicyGradientAgent(p_stay=0., degree_w=1., memory=30,
                                      kernel_func=GaussianKernel(alpha=0.1))

    pos = [0., 0.]
    poses = []

    for i in range(20):
        v = (-pos[0] + pos[1])
        a = model.next_step(v)
        pos[0] += a[0]
        pos[1] += a[1]

        poses.append(model.pos.tolist())

        print(pos, v)

    poses = np.array(poses)
    c = np.linspace(0.2, 0.9, poses.shape[0])
    for i in range(poses.shape[0] - 1):
        plt.plot([poses[i, 0], poses[i + 1, 0]], [poses[i, 1], poses[i + 1, 1]], linewidth=0.5,
                 color=[0., 0.1, 0.4, c[i]])
    plt.scatter(poses[:, 0], poses[:, 1], s=3., c=[[0.1, 0.1, 0.6, i] for i in np.linspace(0.1, 0.9, poses.shape[0])])
    plt.show()

    # model = GradAgent(memory=10, lbd=1e-2, p_stay=0.1, step=1.)
    #
    # pos = [0., 0.]
    # poses = []
    #
    # for i in range(20):
    #     v = (-pos[0] + pos[1])
    #     a = model.next_step(v)
    #     pos[0] += a[0]
    #     pos[1] += a[1]
    #
    #     poses.append(model.pos.tolist())
    #
    #     print(pos, v)
    #
    # poses = np.array(poses)
    # plt.plot(poses[:, 0], poses[:, 1], linewidth=0.5)
    # plt.show()
