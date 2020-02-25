import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class InvertedPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = 1.
        # self.m = 2.
        # self.M = 8.
        self.l = 1.
        # self.a = 1.0/(self.m+self.M)
        # self.l = 1./2.
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        # self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        # M = self.M
        # a = self.a
        l = self.l
        dt = self.dt
        # note: here u is torch not force in the paper
        random_torch = np.random.uniform(-0.2, 0.2)
        # -: left force
        # +: right forces
        if action==0:   u = -1.8 + random_torch
        elif action==1: u = 1.8 + random_torch
        else:           u = 0. + random_torch
        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        # costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        done = False

        if th==0:
            reward = 0
        else:
            reward = -1
            done = True

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        # angle_normalize: [-pi ~ pi)
        newth = angle_normalize(newth)

        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = np.array([newth, newthdot])
        return self._get_obs(), reward, done, {}

    def reset(self):
        # angle_normalize
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        # self.state = np.array([-np.pi/2,0])
        th, thdot = self.state
        th = angle_normalize(th)
        self.state = np.array([th, thdot])
        self.state = np.array([np.pi*1.5,0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    # [-pi ~ pi)
    return (((x+np.pi) % (2*np.pi)) - np.pi)