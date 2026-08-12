import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
keras = json.load(open(os.path.join(HERE, 'keras.json')))
go = json.load(open(os.path.join(HERE, 'go.json')))

kl = np.array(keras['losses'])
gl = np.array(go['losses'])
# Keras 3's train_on_batch returns the sample-weighted running mean of the
# loss since training started, not the loss of that batch. Convert the Go
# per-batch losses to the same running mean for an apples-to-apples check.
sizes = np.array([64, 36] * (len(gl) // 2))
grun = np.cumsum(gl * sizes) / np.cumsum(sizes)
print('step | keras running loss | go running loss | abs diff')
for s in [1, 2, 3, 4, 10, 20, 50, 100, 150, 200]:
    print(f'{s:4d} | {kl[s-1]:.9f}        | {grun[s-1]:.9f}     | '
          f'{abs(kl[s-1]-grun[s-1]):.2e}')


def diffs(a, b, label):
    print(f'-- weights {label}')
    for key in ['w1', 'b1', 'w2', 'b2']:
        x = np.array(a[key], dtype=np.float64)
        y = np.array(b[key], dtype=np.float64)
        d = np.abs(x - y)
        print(f'  {key}: max|d|={d.max():.2e}  mean|d|={d.mean():.2e}  '
              f'max|w|={np.abs(x).max():.3f}')


diffs(keras['after1'], go['after1'], 'after 1 optimizer step')
diffs(keras['final'], go['final'], f'after {len(kl)} optimizer steps')
