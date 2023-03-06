import numpy as np
from vis import *
from game_mgmt import *
from agent import *

def main():
    # game setup
    gridSize = 4
    nPlayers = 2
    S = startGame(gridSize, nPlayers) # state of board is shape(size, size, 2). [x,y,0] is player owner, [x,y,1] is nTroops of cell (x,y)
    A = generateActionSet(gridSize)

    counter = 0
    while True:
        player = np.mod(counter, nPlayers)

        for _ in range(gridSize): # players get a max of gridSize attacks per turn
        
            a = chooseAction(S, player, A)
            # if player hasn't passed their turn
            if a[0] != -1:
                S = doAction(S, player, a)
            # if player has passed their turn
            else: 
                break
            print(S)
            
        
        # visState(S, nPlayers)
        break
    

    return

    

if __name__ == '__main__':
    main()