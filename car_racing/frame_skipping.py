import gym


class FrameSkipping(gym.Wrapper):
    """ Act at every k-th frame instead of every frame. """

    def __init__(self, env, num_skip):

        super().__init__(env)
        self.num_skip = num_skip

    def step(self, action):
        """ Last action is repeated on skipped frames and reward is summed. """

        sum_reward = 0.0

        for _ in range(self.num_skip):

            state, reward, done, info = self.env.step(action)
            sum_reward += reward
            if done:
                break

        return state, sum_reward, done, info
