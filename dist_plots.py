import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr
from half_t import HalfT

def rmse(x, true):
    return np.sqrt(np.nanmean((x-true)**2, axis=0))

def bias(x, true):
    return np.abs(np.nanmean(x, axis=0) - true)

def nan_rate(x):
    return np.isnan(x).sum(axis=0)/x.shape[0]

def cvar_rmse(cvars, cvars_true):
    r = []
    for i in range(len(cvars_true)):
        r1 = rmse(cvars[0,i], cvars_true[i])
        r2 = rmse(cvars[1,i], cvars_true[i])
        r3 = rmse(cvars[2,i], cvars_true[i])
        r.append([r1,r2,r3])
    return np.array(r)

def cvar_bias(cvars, cvars_true):
    b = []
    for i in range(len(cvars_true)):
        b1 = bias(cvars[0,i], cvars_true[i])
        b2 = bias(cvars[1,i], cvars_true[i])
        b3 = bias(cvars[2,i], cvars_true[i])
        b.append([b1,b2,b3])
    return np.array(b)

def cvar_nans(cvars):
    nas = []
    for i in range(cvars.shape[1]):
        n1 = nan_rate(cvars[1,i])
        n2 = nan_rate(cvars[2,i])
        nas.append([n1,n2])
    return np.array(nas)

def get_means(cvars):
    m = np.nanmean(cvars[:,:,:,-1], axis=2)
    f = lambda x: ' & '.join(np.around(x, 2).astype(str))
    m_str = np.apply_along_axis(f, 1, m.transpose())
    for s in m_str:
        print(s)

if __name__ == '__main__':
    cvars_est = np.load('data/cvars.npy')

    # CVaR level
    alph = 0.999

    c = np.array([0.45, 0.6, 0.7, 1.45, 2.75])
    d = np.array([3, 2.5, 2, 1, 0.5])
    dists = [Burr(i,j) for (i,j) in zip(c,d)]
    dists += [Frechet(1.25), Frechet(1.5), Frechet(1.75), Frechet(2), Frechet(2.25), \
             HalfT(1.25), HalfT(1.5), HalfT(1.75), HalfT(2), HalfT(2.25)]

    cvars_est[1][np.where(np.isnan(cvars_est[1]))] = cvars_est[0][np.where(np.isnan(cvars_est[1]))]
    cvars_est[2][np.where(np.isnan(cvars_est[2]))] = cvars_est[1][np.where(np.isnan(cvars_est[2]))]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(10000, 50000, 5).astype(int)


    dist_titles = [d.get_label() for d in dists]

    dist_cvars = [d.cvar(alph) for d in dists]

    dist_rmse = cvar_rmse(cvars_est, dist_cvars)
    dist_bias = cvar_bias(cvars_est, dist_cvars)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=4)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_rows = 6
    n_cols = 5
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(7, 6))

    for i in np.arange(0, n_rows, 2):
        for j in range(n_cols):
            idx = int(i/2)*n_cols+j
            axs[i,j].plot(sampsizes, dist_rmse[idx,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
            axs[i,j].plot(sampsizes, dist_rmse[idx,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
            axs[i,j].plot(sampsizes, dist_rmse[idx,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
            axs[i+1,j].plot(sampsizes, dist_bias[idx,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
            axs[i+1,j].plot(sampsizes, dist_bias[idx,1], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
            axs[i+1,j].plot(sampsizes, dist_bias[idx,2], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
            axs[i,j].set_title(dist_titles[idx])


    #     axs[3,i].set_xlabel('sample size')
    #
    #     axs[0,i].set_title(burr_titles[i])
    #     axs[2,i].set_title(frec_titles[i])
    #
    # axs[0,0].set_ylabel('RMSE')
    # axs[1,0].set_ylabel('absolute bias')
    # axs[2,0].set_ylabel('RMSE')
    # axs[3,0].set_ylabel('absolute bias')
    # axs[0,0].legend(['UPOT', 'BPOT', 'SA'])

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/dist_plots.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
