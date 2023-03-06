import numpy as np
import copy

def startGame(gridSize, nPlayers):
    S = np.zeros((gridSize, gridSize, 2), dtype=int)
    S[:,:,0] = np.random.randint(nPlayers, size=(gridSize, gridSize))     # territory ownership
    S[:,:,1] += 3                                                             # troops per territory
    return S

def get1DState(S):
    gridSize = np.shape(S)[0]
    S_flat = np.zeros((gridSize**2))

    for x in range(gridSize):
        for y in range(gridSize):
            i = np.ravel_multi_index((x,y), (gridSize, gridSize))
            nTroopsBinned = binNTroops(S[x,y,1])
            player = S[x,y,0] + 1
            S_flat[i] = nTroopsBinned * player
    return S_flat

def binNTroops(nTroops):
    maxVal = 6
    if nTroops < maxVal:
        return nTroops
    else:
        return maxVal

def generateActionSet(gridSize):

    # A is list of len nActions with tuple (cellx, celly, attackDirection). attakDirection is north/e/s/w
    # A[0] denotes passing

    A = [(-1,-1,'p')] 

    for x in range(gridSize):
        for y in range(gridSize):
            if x != 0:
                A.append((x,y,'w'))
            if x != gridSize-1:
                A.append((x,y,'e'))
            if y != 0:
                A.append((x,y,'s'))
            if y != gridSize-1:
                A.append((x,y,'n'))

    return A

# check if player p can take action a in game state S
def checkValidAction(S, a, p):
    orig_x = a[0]
    orig_y = a[1]
    if a[2] == 'n':
        dest_y = orig_y + 1
        dest_x = orig_x
    elif a[2] == 's':
        dest_y = orig_y - 1
        dest_x = orig_x
    elif a[2] == 'e':
        dest_y = orig_y
        dest_x = orig_x + 1
    elif a[2] == 'w':
        dest_y = orig_y
        dest_x = orig_x - 1

    doesPlayerOwnOrigin = (S[orig_x, orig_y, 0] == p)
    doesOpponentOwnDestination = (S[dest_x, dest_y, 0] != p)
    doesPlayerHaveEnoughTroops = (S[orig_x, orig_y, 1] > 1)

    if doesPlayerOwnOrigin and doesOpponentOwnDestination:
        return True
    else:
        return False

def doAction(S, attackingPlayer, a):
    S_orig = copy.deepcopy(S)

    # get x,y of defender
    if a[2] == 'n':
        def_y = a[1] + 1
        def_x = a[0]
    elif a[2] == 's':
        def_y = a[1] - 1
        def_x = a[0]
    elif a[2] == 'e':
        def_y = a[1]
        def_x = a[0] + 1
    elif a[2] == 'w':
        def_y = a[1]
        def_x = a[0] - 1

    # attacker can attack with 3 troops at most, defender can defend with 2 troops
    nAttackingTroops = np.minimum(S[a[0], a[1], 1], 3)
    nDefendingTroops = np.minimum(S[def_x, def_y, 1], 2)
    nBattles = np.min((nAttackingTroops, nDefendingTroops))

    # roll a 6-sided die, sort in descending order
    attackRolls = np.sort(np.random.randint(1,7, size=nAttackingTroops))[::-1]
    defendRolls = np.sort(np.random.randint(1,7, size=nDefendingTroops))[::-1]

    # do the battle
    survivingAttackers = attackRolls
    for i in range(nBattles):
        # attacker wins
        if attackRolls[i] > defendRolls[i]: 
            S[def_x, def_y, 1] -= 1

        # defender wins (including ties)
        else: 
            S[a[0], a[1], 1] -= 1
            survivingAttackers -= 1

        # if either side runs out of troops, we're done
        if S[def_x, def_y, 1] == 0 or S[a[0], a[1], 1] == 0:
            break
    
    # after battles complete, check if the territory has been taken
    if S[def_x, def_y, 1] <= 0:
        # set new owner
        S[def_x, def_y, 0] = attackingPlayer 
        # move the attacking troops into the new terrirory 
        S[a[0], a[1], 1] -= survivingAttackers
        S[def_x, def_y, 1] += survivingAttackers
        # additionally, move floor(half) of remaining troops from origin territory into new territory
        troopsToMove = np.floor(S[a[0], a[1], 1] / 2)
        S[a[0], a[1], 1] -= troopsToMove
        S[def_x, def_y, 1] += troopsToMove

    if 1:
        print(f'player {attackingPlayer} attacks from {a[0], a[1]} ({S_orig[a[0], a[1], 1]} troops) into {def_x, def_y} ({S_orig[def_x, def_x, 1]} troops).' + 
              f'\nAttacker rolls {attackRolls}, defender rolls {defendRolls}.' + 
              f'\nAfter attack, origin cell has {S[a[0], a[1], 1]} troops, defending cell has {S[def_x, def_y, 1]} troops,'+
              f'and defending cell is owned by player {S[def_x, def_y, 0]}.')

    return S





