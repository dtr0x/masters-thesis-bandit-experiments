import numpy as np
import os

if __name__ == '__main__':
    arrs = [np.load('data/' + c) for c in os.listdir('data') if 'bandit_cvars_' in c]
    dim = list(arrs[0].shape)
    l = np.sum([a.shape[2] for a in arrs])
    dim[2] = l
    cvars = np.zeros(dim)
    i = 0
    for a in arrs:
        j = a.shape[2]
        cvars[:, :, i:(i+j), :] = a
        i += j
    np.save('data/bandit_cvars.npy', cvars)
