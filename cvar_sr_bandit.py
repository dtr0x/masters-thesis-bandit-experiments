import numpy as np
import os

# argmax resolving ties randonmly
def argmax(arr, axis=0):
    f = lambda x: np.random.choice(np.where(x==x.max())[0])
    return np.apply_along_axis(f, axis, arr)

# parameter for successive rejects sample calculation
def logK(n_arms):
    return 1/2 + (1/np.arange(2, n_arms+1)).sum()

# successive rejects sample size calculation
def bandit_samp_sizes(n_arms, n):
    n_k = lambda k: 1/logK(n_arms) * (n-n_arms)/(n_arms+1-k)
    x = np.apply_along_axis(n_k, 0, np.arange(1, n_arms))
    return np.ceil(x).astype(int)

# successive rejects for one bandit trial
def play(trial):
    n_arms = trial.shape[0]
    n_rounds = trial.shape[1] # = n_arms-1
    arm_idx = np.arange(n_arms)
    for i in range(n_rounds):
        max_arm = argmax(trial[:,i])
        trial = np.delete(trial, max_arm, axis=0)
        arm_idx = np.delete(arm_idx, max_arm)
    return arm_idx.item()

if __name__ == '__main__':
    # CVaR level
    alph = 0.998

    # bandit params
    budgets = np.linspace(10000, 50000, 5).astype(int)
    n_arms = 5
    # the sample sizes for which we will evaluate the cvar
    sampsizes = np.array([bandit_samp_sizes(n_arms, b) for b in budgets]).flatten()

    # load cvar data from file
    cvars = np.load('data/bandit_cvars.npy')

    # run bandit experiments using pre-computed CVaRs
    n_rounds = n_arms-1
    trial_range = np.arange(0, len(sampsizes)+1, n_rounds)
    n_trials = cvars.shape[2]
    arms_selected = np.zeros((len(cvars), len(budgets), n_trials), dtype=np.int32)
    for k in range(len(cvars)):
        for i in range(len(budgets)):
            bandit_data = cvars[k,:,:,trial_range[i]:trial_range[i+1]]
            for j in range(n_trials):
                trial = bandit_data[:,j,:]
                arm_choice = play(trial)
                arms_selected[k, i, j] = arm_choice

    np.save('data/arms_selected.npy', arms_selected)
