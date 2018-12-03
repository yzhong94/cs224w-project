import snap
import numpy as np
from matplotlib import pyplot as plt

### adapted from HW3 Q1.1
### Input: campaign or bill graphs
### Output: +/- lable for each node on which cluster they belong to

def load_graph(name):
    '''
    Helper function to load undirected graphs.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "campaign":
        G = snap.LoadEdgeList(snap.PUNGraph, "../processed-data/campaign_projection.txt", 0, 1)
    elif name == 'bill':
        G = snap.LoadEdgeList(snap.PUNGraph, "../processed-data/bill_projection.txt", 0, 1)
    else:
        raise ValueError("Invalid graph: please use 'campaign' or 'bill'.")
    return G

def get_adjacency_matrix(Graph):
    '''
    This function might be useful for you to build the adjacency matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    #TODO: Your code here
    maxN = 0
    for NI in Graph.Nodes():
        if NI.GetId() > maxN:
            maxN = NI.GetId()

    mat = np.zeros((maxN + 1, maxN + 1))
    
    for EI in Graph.Edges():
        mat[EI.GetSrcNId()][EI.GetDstNId()] = 1
        mat[EI.GetDstNId()][EI.GetSrcNId()] = 1

    return mat
    ##########################################################################

def get_sparse_degree_matrix(Graph):
    '''
    This function might be useful for you to build the degree matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    #TODO: Your code here
    maxN = 0
    for NI in Graph.Nodes():
        if NI.GetId() > maxN:
            maxN = NI.GetId()
    #print "Max index is %d" % (maxN)

    arr = np.zeros(maxN + 1)
    for NI in Graph.Nodes():
        arr[NI.GetId()] = NI.GetOutDeg()

    mat = np.diag(arr)
    '''
    A = get_adjacency_matrix(Graph)
    D = np.zeros((maxN + 1, maxN + 1))
    for i in xrange(0, maxN+1):
        D[i,i] = A[i].sum()
    '''

    return mat
    ##########################################################################

def normalized_cut_minimization(Graph):
    '''
    Implement the normalized cut minimizaton algorithm we derived in the last
    homework here
    '''
    A = get_adjacency_matrix(Graph)
    D = get_sparse_degree_matrix(Graph)
    ##########################################################################
    #TODO: Your code here

    L = np.subtract(D,A)

    #newD = np.linalg.inv(np.sqrt(D))
    newD = 1.0/np.sqrt(D)
    newD[np.isinf(newD)] = 0

    newL = np.matmul(newD,L)
    newL = np.matmul(newL,newD)

    w, v = np.linalg.eigh(newL)

    f = np.matmul(newD, v[:,1]) #fiedler vector
    S = set()
    invS = set()
    for i in range(0, len(f)):
        if f[i] > 0:
            S.add(i)
        else:
            invS.add(i)
    ##TODO: debug this, numbers don't check out yet
    print "Positive Set, S, is size %d" % (len(S))
    print "Negative Set, invS, is size %d" % (len(invS))

    return S, invS
    ##########################################################################

def modularity(Graph, c1, c2):
    '''
    This function might be useful to compute the modularity of a given cut
    defined by two sets c1 and c2. We would normally require sets c1 and c2
    to be disjoint and to include all nodes in Graph
    '''
    ##########################################################################
    #TODO: Your code here
    A = get_adjacency_matrix(Graph)
    twoM = A.sum()
    m = twoM*1.0 /2.0

    cut = 0
    #[EI.GetSrcNId()][EI.GetDstNId()]
    for EI in Graph.Edges():
        if ((EI.GetSrcNId() in c1 and EI.GetDstNId() in c2) 
            or (EI.GetSrcNId() in c2 and EI.GetDstNId() in c1)):
            cut = cut + 1

    volS = 0
    volInvS = 0
    for NI in Graph.Nodes():
        if (NI.GetId() in c1):
            volS = volS + NI.GetOutDeg()
        else:
            volInvS = volInvS + NI.GetOutDeg()

    Q = (-2.0 * cut + volS * volInvS * 1.0 / m) * 1.0 / twoM

    print "Modularity is %f" % (Q)

    return Q
    ##########################################################################

def q1_1():
    '''
    Main workflow
    '''
    ##########################################################################

    #print "Graph: Bill"
    #G = load_graph("bill") ##normal
    #c1, c2 = normalized_cut_minimization(G)

    #Q = modularity(G, c1, c2)
    #print "----------"

    print "Graph: Campaign"
    G = load_graph("campaign") ##normal
    c1, c2 = normalized_cut_minimization(G)
    output(c1, c2)

    Q = modularity(G, c1, c2)
    print "----------"


    pass
    ##########################################################################

def output(s1, s2):
    s1 = list(s1)
    s2 = list(s2)

    res = np.zeros((len(s1)+len(s2), 2), int)
    for i in xrange(0,len(s1)):
        res[i][0] = s1[i]
        res[i][1] = 1

    for j in xrange(0,len(s2)):
        res[j + len(s1)][0] = s2[j]
        res[j + len(s1)][1] = -1

    print "Saving csv file..."
    np.savetxt('spectral.out', res, delimiter=',')

    return

if __name__ == "__main__":

    q1_1()

