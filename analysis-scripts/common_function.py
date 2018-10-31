import snap

def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    X, Y = [], []
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)

    for item in DegToCntV:
        X.append(item.GetVal1()) #get degree
        Y.append(item.GetVal2()/float(Graph.GetNodes())) #get proportion of nodes with certain degree
    ############################################################################
    return X, Y


def calcClusteringCoefficientSingleNode(Node, Graph):
    """
    :param - Node: node from snap.PUNGraph object. Graph.Nodes() will give an
                   iterable of nodes in a graph
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: local clustering coeffient of Node
    """
    ############################################################################
    # TODO: Your code here!
    C = 0.0
    deg = []

    neighbors = snap.TIntV()
    
    ki = Node.GetOutDeg() #get the outer degree of the node
        
    if ki >= 2: #if outer degree is greater than or equal to two
        #reset variables
        ei = 0 
        Subgraph = snap.TUNGraph.New()

        for N in Node.GetOutEdges(): #store all neighbors in neighbors
            neighbors.Add(N)
        
        SubGraph = snap.GetSubGraph(Graph, neighbors) #create a sub graph of neighbors only
        
        ei = SubGraph.GetEdges() #ei is the number of edges between neighbors

        C = 2*abs(ei)/float(ki*(ki-1))
    else:
        C = 0
    ############################################################################
    
    return C

def calcClusteringCoefficient(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: clustering coeffient of Graph
    """
    ############################################################################
    # TODO: Your code here! If you filled out calcClusteringCoefficientSingleNode,
    #       you'll probably want to call it in a loop here
    C = 0.0
    Ci = []

    Nodes = Graph.Nodes()
    for node in Nodes:
        Ci.append(calcClusteringCoefficientSingleNode(node,Graph))
        #print calcClusteringCoefficientSingleNode(node,Graph)
    
    C = 1/float(len(Ci))*float(sum(Ci))

    ############################################################################
    return C


def main():


    pass


if __name__ == "__main__":
    main()