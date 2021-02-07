import argparse
import numpy as np
from run_sim import get_cvars
from cvar_sr_bandit import bandit_samp_sizes

# pre-compute CVaRs for bandit experiments
def run_worker(alph, sampsizes, row_start, row_end):
    data = np.load('data/samples.npy')[:, row_start:row_end, :]
    cvars_est = get_cvars(data, alph, sampsizes)
    cvars_est[1][np.where(np.isnan(cvars_est[1]))] = cvars_est[0][np.where(np.isnan(cvars_est[1]))]
    cvars_est[2][np.where(np.isnan(cvars_est[2]))] = cvars_est[1][np.where(np.isnan(cvars_est[2]))]

    cvars = cvars_est[:3]
    burr_cvars = cvars[:,:5,:,:]
    frec_cvars = cvars[:,5:10,:,:]
    t_cvars = cvars[:,10:,:,:]

    # SA CVaRs at indices 0, 3, 6
    # BPOT CVaRs at indices 1, 4, 7
    # UPOT CVaRs at indices 2, 5, 8
    cvars_all = np.vstack([burr_cvars, frec_cvars, t_cvars])
    np.save('data/bandit_cvars_{}_{}.npy'.format(row_start, row_end), cvars_all)
    return cvars_all

if __name__ == '__main__':
    # get arguments from command line for asset type and year range
    parser = argparse.ArgumentParser()
    parser.add_argument('-row_start', type=int, required=True)
    parser.add_argument('-row_end', type=int, required=True)
    args = parser.parse_args()
    row_start = args.row_start
    row_end = args.row_end

    # CVaR level
    alph = 0.998

    # bandit params
    budgets = np.linspace(10000, 50000, 5).astype(int)
    n_arms = 5
    # the sample sizes for which we will evaluate the cvar
    sampsizes = np.array([bandit_samp_sizes(n_arms, b) for b in budgets]).flatten()

    # compute CVaR data
    cvars = run_worker(alph, sampsizes, row_start, row_end)
