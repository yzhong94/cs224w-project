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

def getGraph(df):
    G = snap.TUNGraph.New()

    for index, row in df.iterrows():
        SrcNId = int(row['SrcNId'])
        DstNId = int(row['DstNId'])
        if G.IsNode(SrcNId) == False:
            G.AddNode(SrcNId)
        if G.IsNode(DstNId) == False:
            G.AddNode(DstNId)
        G.AddEdge(SrcNId, DstNId)

    return G

def getCoSponsor(G, bill_node,legislator_node):
    '''
    returns the one mode projection graph of co-sponsorship

    '''
    CoSponsor = snap.TUNGraph.New()

    print len(legislator_node)
    for i in range(len(legislator_node)):
        for j in range(i+1,len(legislator_node)):
            Nbrs = snap.TIntV()
            if snap.GetCmnNbrs(G,legislator_node[i],legislator_node[j]) != 0:
                if CoSponsor.IsNode(legislator_node[i]) == False:
                    CoSponsor.AddNode(legislator_node[i])
                if CoSponsor.IsNode(legislator_node[j]) == False:
                    CoSponsor.AddNode(legislator_node[j])
                if CoSponsor.IsEdge(legislator_node[i],legislator_node[j]) == False:
                    CoSponsor.AddEdge(legislator_node[i],legislator_node[j]) 

    #snap.SaveEdgeList(CoSponsor, 'cosponsor.txt')

    return CoSponsor


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