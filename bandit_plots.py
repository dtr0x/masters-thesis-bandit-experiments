import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr
from half_t import HalfT

def error_prob(result, best_arm):
    n_trials = result.shape[1]
    return 1 - np.asarray(result == best_arm).sum(axis=1)/n_trials

if __name__ == '__main__':
    c = np.array([0.4, 0.5, 0.75, 1.5, 2.5])
    d = np.array([4, 3, 2, 1, 0.6])
    burr_dists = [Burr(i,j) for (i,j) in zip(c,d)]
    frec_dists = [Frechet(1.25), Frechet(1.5), Frechet(1.75), Frechet(2), Frechet(2.25)]
    t_dists = [HalfT(1.25), HalfT(1.5), HalfT(1.75), HalfT(2), HalfT(2.25)]

    # sample sizes to test CVaR estimation
    budgets = np.linspace(10000, 50000, 5).astype(int)

    # CVaR level
    alph = 0.998

    # get bandit experiment results
    arms_selected = np.load('data/arms_selected.npy')

    # get best arms
    burr_optim = np.argmin([d.cvar(alph) for d in burr_dists])
    frec_optim = np.argmin([d.cvar(alph) for d in frec_dists])
    t_optim = np.argmin([d.cvar(alph) for d in t_dists])

    # error probability metrics
    burr_sa = error_prob(arms_selected[0], burr_optim)
    burr_bpot = error_prob(arms_selected[1], burr_optim)
    burr_upot = error_prob(arms_selected[2], burr_optim)
    frec_sa = error_prob(arms_selected[3], frec_optim)
    frec_bpot = error_prob(arms_selected[4], frec_optim)
    frec_upot = error_prob(arms_selected[5], t_optim)
    t_sa = error_prob(arms_selected[6], t_optim)
    t_bpot = error_prob(arms_selected[7], t_optim)
    t_upot = error_prob(arms_selected[8], t_optim)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=5)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5))

    # Burr plots
    axs[0].plot(budgets, burr_sa, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[0].plot(budgets, burr_bpot, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[0].plot(budgets, burr_upot, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    axs[0].set_title('Burr bandit')

    # Frechet plots
    axs[1].plot(budgets, frec_sa, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[1].plot(budgets, frec_bpot, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[1].plot(budgets, frec_upot, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    axs[1].set_title('Frechet bandit')

    # T plots
    axs[2].plot(budgets, t_sa, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[2].plot(budgets, t_bpot, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[2].plot(budgets, t_upot, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='k')
    axs[2].set_title('half-t bandit')

    for i in range(3):
        axs[i].set_xlabel('budget')
        axs[i].legend(['SA', 'BPOT', 'UPOT'])
        axs[i].set_xticks(budgets)
        axs[i].ticklabel_format(axis='x', style='sci', scilimits=(0,0))


    plt.tight_layout(pad=0.5)
    axs[0].set_ylabel('probability of error')
    fig.savefig('plots/bandit_plots.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
