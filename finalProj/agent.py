import numpy as np
import random
from game_mgmt import *

# choose an action (x, y, direction) from action set A in board state S in using random or deep q-learning
def chooseAction(S, p, A, model=None):
    gridSize = np.shape(S)[0]
    randomAgentLikelihoodOfPassing = 1 # percent
    
    if model == None: # choose randomly
        # the random agent has a 20% change of just passing and not attacking
        if np.random.randint(100) < randomAgentLikelihoodOfPassing:
            return A[0]
        else:
            actionList = np.arange(len(A))
            random.shuffle(actionList)
    else:  # use deep Q-learning
        S_flat = get1DState(S)
        # TODO pass S_flat to deepQ to get action_probs
        pass

    # iterate through prioritized action list, pick the first valid action
    for a_idx in actionList:
        if checkValidAction(S, A[a_idx], p):
            return A[a_idx]


def chooseTroopPlacement(S, p, model=None):
    gridSize = np.shape(S)[0]
    if model == None: 
        placementList = np.arange(gridSize**2)
        random.shuffle(placementList)
    else:  # use deep Q-learning
        S_flat = get1DState(S)
        # TODO pass S_flat to deepQ to get action_probs
        pass
    
    for i in range(len(placementList)):
        cellToPlace = placementList[i]
        cellxy = np.unravel_index(cellToPlace, (gridSize, gridSize))

        # if player owns the territory
        if p == S[cellxy[0], cellxy[1], 0]:
            break
    
    S[cellxy[0], cellxy[1], 1] += 1
    return S
 
