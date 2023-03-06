import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visState(S, nPlayers):
    gridSize = np.shape(S)[0]
    plt.imshow(S[:,:,0], alpha=0.5, origin='lower')
    for x in range(gridSize):
        for y in range(gridSize):
            plt.text(x,y,S[x,y,1])
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.show()