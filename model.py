import pandas as pd
import argparse
import math
import pickle
import os
import numpy as np
import dill
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from bayes_opt import BayesianOptimization

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("x", help="run in train mode", type=float,  nargs='?')
parser.add_argument("z", help="run in train mode", type=float,  nargs='?')
parser.add_argument("alpha", help="run in train mode", type=float, nargs='?')
parser.add_argument("--mode", help="run in train mode", type=int)
parser.add_argument("--input_path", help="run in train mode", type=str)
parser.add_argument("--output_path", help="run in train mode", type=str)
parser.add_argument("--mass", help="run in train mode", type=int)
parser.add_argument("--y", help="run in train mode", type=int)
parser.add_argument("--step_size", help="run in train mode", type=int)
parser.add_argument("--values", help="run in train mode", type=str)


class Model:
    def __init__(self):
        self.speed = None
        self.force = None
        self.mass = None
        self.y = None

    def train(self, F, Wind):
        self.pf = PolynomialFeatures(degree=3)
        self.lr = LinearRegression()
        X_train = np.array(F.V)[:, np.newaxis]
        self.pf.fit(X_train)
        self.lr.fit(self.pf.transform(X_train), F.Fa + 10e-10)

        self.svr_x = SVR(gamma=0.0001)
        self.svr_x.fit(np.array(Wind.Y)[:, np.newaxis], np.array(Wind.Wx)[:, np.newaxis])

        self.svr_z = SVR(gamma=0.0001)
        self.svr_z.fit(np.array(Wind.Y)[:, np.newaxis], np.array(Wind.Wz)[:, np.newaxis])
        self.speed = lambda y: (self.svr_x.predict([[y]])[0], self.svr_z.predict([[y]])[0])
        self.force = lambda v: np.sign(v) * (self.lr.predict(self.pf.transform([[abs(v)]]))[0] if v > 10 else 1.4 * abs(v))
        return self

    def coordinates(self, x, z, alpha):
        dy = -100
        m = 100
        y = 1400
        w_x, w_z = self.speed(y)
        g = -9.81
        v = 250
        v_y = 0
        v_x = v * math.cos(alpha)
        v_z = v * math.sin(alpha)
        a_y = g
        while y>0:
            p = [a_y/2, v_y, y-dy]
            t_1_2 = np.roots(p)
            if t_1_2[0]>t_1_2[1]:
                t = t_1_2[0]
            else:
                t = t_1_2[1]
            a_x = (-self.force(v_x) + self.force(w_x))/m
            a_z = (-self.force(v_z) + self.force(w_z))/m
            dx = x + v_x * t + a_x * t**2 / 2
            dz = z + v_z * t + a_z * t**2 / 2
            y = y + dy
            x = x + dx
            z = z + dz
            v_y = v_y + a_y * t
            v_x = v_x + a_x * t
            v_z = v_z + a_z * t
            w_x, w_z = self.speed(y)
        s = math.sqrt(x**2 + z**2)
        return s

if __name__ == "__main__":
    args = parser.parse_args()
    mode = args.mode
    print(args)
    if mode == 0:
        F = pd.read_csv(os.path.join(args.input_path, 'F.csv'), sep=";")
        Wind = pd.read_csv(os.path.join(args.input_path, 'Wind.csv'), sep=";")
        model = Model()
        model.train(F, Wind)
        with open(os.path.join(args.output_path, 'model.pickle'), "wb") as f:
            dill.dump(model, f)

    if mode == 1:
        with open(os.path.join(args.input_path), "rb") as f:
            model = dill.load(f)
        model.step_size = args.step_size
        model.mass = args.mass
        model.y = args.y
        print(model.coordinates(args.x, args.z, args.alpha))

    if mode == 2:
        with open(os.path.join(args.input_path), "rb") as f:
            model = dill.load(f)
        model.step_size = args.step_size
        model.mass = args.mass
        model.y = args.y
        pbounds = {'x': (0, 5), 'z': (0, 5), 'alpha': (0, 2 * np.pi)}
        optimizer = BayesianOptimization(
            pbounds=pbounds,
            f=lambda x, z, alpha: -model.coordinates(x, z, alpha),
            random_state=42,
        )
        
        optimizer.maximize()
        result = optimizer.max
        print(result["target"])
        print(result["params"])
