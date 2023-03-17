import numpy as np
import time
from vis import *
from game_mgmt import *
from agent import *
import timeit




def main():
    ######################
    ###### settings ######
    ######################
    visType = 'end' # 'end', 'each', or 'none'
    gridSize = 4
    nPlayers = 2
    ######################
    
    # game setup
    np.random.seed(1)
    maxTurns = 250
    maxAttacksPerTurn = 5
    S = startGame(gridSize, nPlayers) # state of board is shape(size, size, 2). [x,y,0] is player owner, [x,y,1] is nTroops of cell (x,y)
    A = generateActionSet(gridSize)

    # housekeeping
    if visType == 'end':
            clearAllPlots()
    turnCount = 0
    S_history = []
    figs = []
    startt = timeit.default_timer()
    
    for turnCount in range(maxTurns):
        runTime = (timeit.default_timer() - startt) / 60
        print(f'turn {turnCount} ({round((turnCount/runTime),2)} turns/min)', end="\r")
        player = np.mod(turnCount, nPlayers)
        
        figs = visState(S, nPlayers, visType, figs, title=f'Turn {turnCount}: Start of player {player}\'s turn')
        
        nTroopsToPlace = getResupplyCount(S, player)
        for t in range(nTroopsToPlace):
            S = chooseTroopPlacement(S, player)
        figs = visState(S, nPlayers, visType, figs, title=f'Turn {turnCount}: Player {player} troops placed')

        for _ in range(maxAttacksPerTurn): # players get a max of gridSize attacks per turn
            S_history.append(copy.deepcopy(S))
            S_orig = copy.deepcopy(S)
            a = chooseAction(S, player, A)
            # if player hasn't passed their turn
            if a[0] != -1:
                S, rolls = doAction(S, player, a)
            # if player has passed their turn
            else: 
                figs = visState(S_orig, nPlayers, visType, figs,title=f'Turn {turnCount}: Player {player} ends their turn')
                break
            title = f'Turn {turnCount}: Player {player} attacks {a[2]} from {a[0], a[1]} \n Attacker rolls {rolls[0]}, Defender rolls {rolls[1]}'
            figs = visState(S_orig, nPlayers, visType, figs, title=title, action=a)
            plt.close()
            figs = visState(S, nPlayers, visType, figs, title=f'Turn {turnCount}: Player {player}\'s attack outcome')
            plt.close()
        
        # Check if game is complete
        if np.all(S[:,:,0] == S[0,0,0]):
            title=f'Turn {turnCount}: End of game! Player {S[0,0,0]} wins!'
            figs = visState(S, nPlayers, visType, figs, title=title)
            break

    if visType == 'end':
        visGameFlow(figs)
    plotGameProgress(S_history, nPlayers)
    print('\n Game complete!')
    return
    

if __name__ == '__main__':
    main()