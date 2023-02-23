import numpy as np
import pandas
import matplotlib.pyplot as plt
import copy
from datetime import datetime

def sarsalambda(D, discount,maxRuntime_mins=10):
    # (0) define constants, adjust for 0-indexing, get metadata
    gamma = 0.9     # trace decay rate
    alpha = 0.01    # learning rate
    D.s, D.a, D.sp = D.s-1, D.a-1, D.sp-1
    nExs, nStates, nActions = len(D.s[:]), max(D.s[:])+1, max(D.a[:])+1

    # (1) initialize Q(s,a) and N(s,a)
    Q = np.zeros((nStates, nActions))
    Q_prev = np.zeros((nStates, nActions))
    N = np.zeros((nStates, nActions), dtype=np.uint64)

    counter = 0
    startt = datetime.now()
    checkInterval = 5
    convergence = {'t':[], 'change':[]}

    while True:
        i = np.mod(counter, nExs)

        # (2) get s, a, r, s'
        s, a, r, sp = D.s[i], D.a[i], D.r[i], D.sp[i]

        # (3) get action a' in state s'
        ap = np.argmin(Q[sp,:])

        # (4) sarsa difference update: delta <- r + gamma * Q(s',a') - Q(s,a)
        delta = r + gamma * Q[sp, ap] - Q[s,a]

        # (5) state-action count update: N(s,a) <- N(s,a) + 1
        N[s,a] += 1

        for s in range (nStates):
            for a in range(nActions):
                # (6) Update Q: Q(s,a) <- Q(s,a) + alpha * delta * N(s,a)
                Q[s,a] = Q[s,a] + alpha * delta * N[s,a]
                # (7) Decay counts: N(s,a) <- N(s,a) * gamma * lambda
                N[s,a] = N[s,a] * gamma * discount
        
        # (8) Stop if Q change is slow or max time elapsed
        if np.mod(counter, checkInterval) == 0 and counter > checkInterval:
            runTime_mins = np.round((datetime.now() - startt).total_seconds()/60, 2)
            pctChange = 100*(np.mean(np.abs((Q-Q_prev) / Q_prev)))
            convergence['t'].append(runTime_mins)
            convergence['change'].append(pctChange)
            print(f'runtime {runTime_mins} mins: Q at count {counter} is {pctChange.round(1)} % different than Q at count {counter-checkInterval}')

            if pctChange < 0.1 or runTime_mins > maxRuntime_mins:
                break
            else:
                Q_prev = copy.deepcopy(Q)
        counter += 1

    # (9) Find optimal policy
    pi = np.zeros(nStates)
    for s in range(nStates):
        pi[s] = np.argmax(Q[s,:])

    return pi+1, convergence
# lookahead is R(s,a) + lambda * sum_{s'}[T(s'|s,a) * U_{k}(s')
def lookahead(R, T, U, discount, s, a):
    nStates = np.shape(R)[0]
    futureReward = 0
    for sp in range(nStates):
        futureReward += T[s, a, sp] * U[sp]
    return R[s, a] + discount * futureReward

# using Gauss-Seidel value iteration
def valueIter(R, T, discount, maxRuntime_mins=10):
    nStates, nActions = np.shape(R)[0], np.shape(R)[1]
    actionSet = range(nActions)
    U = np.zeros((nStates))
    U_prev = np.zeros((nStates))

    startt = datetime.now()
    checkInterval = 5

    for b in range(int(1E6)):
        for s in range(nStates):
            U[s] = max([lookahead(R, T, U, discount, s, a) for a in actionSet])

        if np.mod(b, checkInterval) == 0 and b > checkInterval:
            runTime_mins = np.round((datetime.now() - startt).total_seconds()/60, 2)

            pctChange = 100*(np.mean(np.abs((U-U_prev) / U_prev)))
            # print(f'runtime {runTime_mins} mins: U at backup {b} is {pctChange.round(1)} % different than U at backup {b-checkInterval}')

            if pctChange < 0.1 or runTime_mins > maxRuntime_mins:
                break
            else:
                U_prev = copy.deepcopy(U)

    return U

def maxLikelihoodModel(D, discount, maxRuntime_mins):
    # (0) change D so everything starts at 0
    D.s, D.a, D.sp = D.s-1, D.a-1, D.sp-1

    # (1) get counds N(s,a,s')
    nExs, nStates, nActions = len(D.s[:]), max(D.s[:])+1, max(D.a[:])+1
    N = np.zeros((nStates, nActions, nStates), dtype=np.uint16)
    for i in range(nExs):
        N[D.s[i], D.a[i], D.sp[i]] += 1

    # (2) get transition model T(s'|s,a) = N(s,a,s') / N(s,a)
    T = N / np.sum(N, axis=2)[:, :, None]

    # (3) get reward model R(s,a) = rho(s,a) / N(s,a) where rho(s,a) is sum of all rewards taking action a in state s
    rho = np.zeros((nStates, nActions))
    for i in range(nExs):
        rho[D.s[i], D.a[i]] += D.r[i]
    R = rho / np.sum(N, axis=2)

    # (4) perfom bellman backup U_{k+1}(s) = max_{a}[R(s,a) + lambda * sum_{s'}[T(s'|s,a) * U_{k}(s')]]
    U = valueIter(R, T, discount, maxRuntime_mins)

    # (5) get policy pi(s) = argmax_{a}(U(s))
    pi = np.zeros(nStates)
    actionSet = range(nActions)
    for s in range(nStates):
        pi[s] = np.argmax([lookahead(R, T, U, discount, s, a)
                          for a in actionSet])

    return pi+1  # change to 1- indexing

def main():
    # settings
    name = 'medium'
    maxRuntime_mins = 10

    startt = datetime.now()

    # problem info
    infile = "data/" + name + ".csv"
    outfile = "policies/" + name + ".policy"
    discounts = {"small": 0.95, "medium": 1, "large": 0.95}
    discount = discounts[name]

    D = pandas.read_csv(infile)

    if name == 'small':
        policy = maxLikelihoodModel(D, discount, maxRuntime_mins)
    else:
        policy, convergence = sarsalambda(D, discount, maxRuntime_mins)

    np.savetxt(outfile, policy, fmt='%i', delimiter=',')
    runTime_mins = np.round((datetime.now() - startt).total_seconds()/60, 2)
    print(f'saved policy to: \"{outfile}\", runtime {runTime_mins}')

    if 1:
        plt.plot(convergence['t'], convergence['change'])
        plt.show()

if __name__ == '__main__':
    main()
