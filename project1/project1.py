import sys
import numpy as np
import networkx as nx
import pandas
import matplotlib.pyplot as plt
import scipy.special
import copy
from datetime import datetime
from networkx.drawing.nx_pydot import graphviz_layout

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def lg(x):
    return scipy.special.loggamma(x)

def bayes_score_component(M, alpha):
    
    p = np.sum(lg(alpha + M))
    p = p - np.sum(lg(alpha))
    p = p + np.sum(lg(np.sum(alpha, axis=1)))
    p = p - np.sum(lg(np.sum(alpha, axis=1) + np.sum(M, axis=1)))

    return p

def getIdxFromParentState(r, idxOfParents, stateOfParents):
    # r = num instantiations of each variable
    returnIdx = 0

    nStatesForEachParent = [r[p] for p in idxOfParents]
    q = np.prod(nStatesForEachParent) # total number of parent states
    idxOfAllParentStates = range(q)
    idxOfAllParentStates = np.reshape(idxOfAllParentStates, (nStatesForEachParent))
    return idxOfAllParentStates[tuple(stateOfParents)]

def counts(vars, G, D, graphInfo):
    # D = data of shape(n,m) where n is num variables and m is num datapts
    # G = bayes struct
    # returns array M of shape(n,)

    # n = num variables
    n = np.shape(D)[0] 
    m = np.shape(D)[1]

    # r = num instatiations of var[i]
    r = graphInfo['r']
    # q = num instantations of var[i]'s parents
    q = graphInfo['q']
    # m = list of size n, each element containing a shape(q[i], r[i]) containing counts of each state
    M = [np.zeros((q[i], r[i])) for i in range(n)]

    for datapt in range(m):
        for datapt_item in range(n):
            state_of_self = D[datapt_item, datapt]
            parents = list(G.predecessors(datapt_item))
            state_of_parents_idx = 0
            if len(parents) != 0:
                state_of_each_parent = [D[parent, datapt] for parent in parents]
                state_of_parents_idx = getIdxFromParentState(r, parents, state_of_each_parent)

            M[datapt_item][state_of_parents_idx, state_of_self] += 1
    return M
            

def prior(vars, graphInfo):
    n = len(vars)
    r = graphInfo['r']
    q = graphInfo['q']
    
    # m = list of size n, each element containing a shape(q[i], r[i]) containing counts of each state
    M = [np.ones((q[i], r[i])) for i in range(n)]
    return M



def bayes_score(vars, G, D, graphInfo):
    n = np.shape(vars)[0]
    M = counts(vars, G, D, graphInfo)
    alpha = prior(vars, graphInfo)
    return np.sum([bayes_score_component(M[i], alpha[i]) for i in range(n)])

def getGraphInfo(n, G, D):
    # r = num instatiations of var[i]
    r = [np.max(D[i])+1 for i in range(n)]
    # q = num instantations of var[i]'s parents
    q = [int(np.prod([r[j] for j in list(G.predecessors(i))])) for i in range(n)]

    returnVal = {'r': r, 'q': q}
    return returnVal

def compute(infile, outfile):
    startt=datetime.now()

    # (1) read in data
    D = pandas.read_csv(infile)
    vars = D.columns
    D = np.array(D).T - 1 # subtract 1 for 0-indexing
    n = len(vars)

    # (2) get a graph, set nodes to 0, 1, ..., n
    G = nx.DiGraph()
    G.add_nodes_from(range(n)) 
    graphInfo = getGraphInfo(n, G, D) 
    
    # (3) score the graph
    score = bayes_score(vars, G, D, graphInfo)
    allTimeBest = {'G': G, 'score': score}

    # (4) optimize the graph using local directed graph search
    maxTime = 20 # mins
    maxAttempts = 1000
    undoRate = np.floor(maxAttempts/10)
    resetRate = np.floor(maxAttempts/5)
    consecutiveNewGraphsWithoutImprovement = 0
    consecutiveNewGraphsWithoutAllTimeImprovement = 0
    totalGraphsGenerated = 0
    scores = np.array([[0, score]])
    mostRecentChange = {'type': None, 'nodes': None}
    while True:

        runtime_mins = np.round((datetime.now() - startt).total_seconds()/60, 2)
        G_new = copy.deepcopy(G)
        existingEdges = list(G_new.edges)
        # (4.1) add an edge
        decision = np.random.randint(0,3)
        if decision == 0 or len(existingEdges) == 0: 

            for i in range(maxAttempts):
                # pick start and end node, make sure they're not the same and don't already exist
                for j in range(maxAttempts):
                    startNode = np.random.randint(0, n)
                    endNode = np.random.randint(0, n)
                    if startNode != endNode and (startNode, endNode) not in list(G.edges) and (endNode, startNode) not in list(G.edges):
                        break

                # add to graph. If we still have a DAG, we can go get the score
                G_new.add_edge(startNode, endNode)
                
                if nx.is_directed_acyclic_graph(G_new):
                    break
        
        # (4.2) alternatively, remove or flip and existing edge
        else:
            
            for i in range(maxAttempts):
                existingEdges = list(G_new.edges)
                edgeToChange = existingEdges[np.random.randint(0, len(existingEdges))]
                startNode = edgeToChange[0]
                endNode = edgeToChange[1]
                if decision == 1: # remove
                    G_new.remove_edge(startNode, endNode)
                else: # flip
                    G_new.remove_edge(startNode, endNode)
                    G_new.add_edge(endNode, startNode)
                if nx.is_directed_acyclic_graph(G_new):
                    break

        # (4.3) get the score of G_new, and update G if the score is an improvement
        if nx.is_directed_acyclic_graph(G_new):
            graphInfo_new = getGraphInfo(n, G_new, D) 
            score_new = bayes_score(vars, G_new, D, graphInfo_new)
            totalGraphsGenerated += 1

            if score_new > score or np.random.randint(0, 20) == 0:
                if score_new > score:
                    consecutiveNewGraphsWithoutImprovement = 0
                if score_new > allTimeBest['score']:
                    improvement = round(score_new-allTimeBest['score'],2)
                    print(f'New all time best score! {round(score_new,2)} (change: {improvement})')
                    allTimeBest['score'] = score_new
                    allTimeBest['G'] = G_new
                    consecutiveNewGraphsWithoutAllTimeImprovement = 0
                    
                G = copy.deepcopy(G_new)
                score = copy.deepcopy(score_new)
                scores = np.append(scores, np.array([[runtime_mins,score]]), axis=0)
                mostRecentChange['type'] = decision
                mostRecentChange['nodes'] = [startNode, endNode]
                print(f'Improved score {round(score,2)}. Runtime = {runtime_mins}')
            else:
                consecutiveNewGraphsWithoutImprovement += 1
                consecutiveNewGraphsWithoutAllTimeImprovement += 1
        else: 
            consecutiveNewGraphsWithoutImprovement += 1
            consecutiveNewGraphsWithoutAllTimeImprovement += 1

        # (4.4) print status 
        if np.mod(consecutiveNewGraphsWithoutAllTimeImprovement, 100) == 0 and consecutiveNewGraphsWithoutAllTimeImprovement > 99:
            print(f'consective graphs without all-time score improvement  = {consecutiveNewGraphsWithoutAllTimeImprovement}' +
            f' runtime = {runtime_mins}')

        # (4.5) if no changes seem to help, undo the last change
        if np.mod(consecutiveNewGraphsWithoutImprovement, undoRate) == undoRate-1:
            print(f'No changes are helping - removing last change')
            if mostRecentChange['type'] == 0: # undo an add
                try:
                    G.remove_edge(mostRecentChange['nodes'][0], mostRecentChange['nodes'][1])
                except:
                    print(f'error, cannot remove nodes from graph')
            elif mostRecentChange['type'] == 1: # undo a remove
                try:
                    G.add_edge(mostRecentChange['nodes'][0], mostRecentChange['nodes'][1])
                except:
                    print(f'error, cannot add nodes to graph')
            elif mostRecentChange['type'] == 2: # undo a flip
                try:
                    G.remove_edge(mostRecentChange['nodes'][1], mostRecentChange['nodes'][0])
                    G.add_edge(mostRecentChange['nodes'][0], mostRecentChange['nodes'][1])
                except:
                    print(f'error, cannot flip nodes')
        
        
        if np.mod(consecutiveNewGraphsWithoutAllTimeImprovement, resetRate) == resetRate-1:
            print(f'{resetRate} consecutive graphs w/o all-time improvement - resetting back to all-time G')
            G = copy.deepcopy(allTimeBest['G'])
            
        
        # (4.6) if we've had too many consecutive duds, we're done
        if consecutiveNewGraphsWithoutImprovement > maxAttempts/10 or consecutiveNewGraphsWithoutAllTimeImprovement > maxAttempts or runtime_mins > maxTime:
            break

    # (5) return the graph when done
    network_str = ''
    G = copy.deepcopy(allTimeBest['G'])
    
    for i in range(len(vars)):
        parentName = vars[i]
        kidsIdxs = list(G.successors(i))
        if len(kidsIdxs) != 0:
            for k in kidsIdxs:
                strToAdd = f'{parentName}, {vars[k]} \n'
                network_str += strToAdd
        else:
            network_str += f'{parentName}, \n'
    graphsPerMin = round(totalGraphsGenerated/runtime_mins, 2)
    print(f'generated {totalGraphsGenerated} graphs in {runtime_mins} mins ({graphsPerMin} graphs/min), best score = {score}')
    print(network_str)

    if 1:
        plt.figure()
        plt.plot(scores[:,0], scores[:,1])
        plt.xlabel('time (mins)')
        plt.ylabel('graph bayesian score')
        plt.title(f'{infile} score vs time')

        nodeMapping = dict(zip(list(range(len(vars))), list(vars)))
        G = nx.relabel_nodes(G, nodeMapping)
        options = {
            "font_size": 10,
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "gray",
            "linewidths": 5,
            "width": 5,
        }
        plt.figure()
        # pos = nx.shell_layout(G)
        pos = graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos=pos, arrows=True, arrowstyle='->', **options)
        plt.title(f'{infile} graph')
        plt.axis("off")
        plt.show()
    print('done')

def main():
    
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    if 0:
        main()
    else:
        inputfilename = "data/large.csv"
        outputfilename = "graphs/large.gph"
        compute(inputfilename, outputfilename)
