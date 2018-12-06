import snap
import pandas as pd
import bill
import common_function
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
import networkx
import time
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif

def loadBillData(term):
    '''
    For a given term, load the bill data (bill - candidate) and a pd dataframe with 
    SrcNId: bill node #, DstNId: legislator node
    '''
    legislator_bill = pd.read_csv('processed-data/legislator_bill_edge_list.csv')

    term_legislator_bill = legislator_bill[legislator_bill['congress_term'] == term]
    return term_legislator_bill[['SrcNId','DstNId']]

def loadFinancialData(start_year, end_year):
    '''
    For a given year range, load the financial data (candidate - committee) and a pd dataframe with 
    SrcNId: comm node #, DstNId: legislator node
    '''
    financial_data = pd.read_csv('processed-data/campaignNetworks_raw.csv')
    term_financial_data = financial_data[(financial_data['ContributionYear'] >= start_year) & (financial_data['ContributionYear'] <= end_year )]

    #print financial_data[financial_data['lname'] == "LEWIS"]['fname'].unique()

    term_financial_data = term_financial_data.rename(index=str, columns={"NodeID": "DstNId", "ComNodeId": "SrcNId"})
    print list(term_financial_data)
    return term_financial_data[['DstNId','SrcNId','TRANSACTION_AMT']]

def loadParty(path):
    party_data = pd.read_csv(path)
    #print list(party_data)
    party_data = party_data[['NodeID','Party']]
    party_data = party_data.drop_duplicates(subset = ['NodeID','Party'])
    #print party_data
    return party_data

def getSponsorLink(df):
    '''
    For a given term's df, return the cosponsor projection in snap undirected graph
    '''
    bill_node = df['SrcNId'].unique()   
    legislator_node = df['DstNId'].unique()

    G = common_function.getGraph(df)

    CoSponsor = common_function.getCoSponsor(G, bill_node,legislator_node)

    return CoSponsor

def getCampaign(df):
    '''
    For a given term's campaign data, return a bi-partite graph with edge attributes of contribution amount
    '''
    G_Campaign = snap.TNEANet.New()
    G_Campaign.AddIntAttrE('TRANSACTION_AMT',0)

    for index, row in df.iterrows():
        SrcNId = int(row['SrcNId'])
        DstNId = int(row['DstNId'])
        if G_Campaign.IsNode(SrcNId) == False:
            G_Campaign.AddNode(SrcNId)
        if G_Campaign.IsNode(DstNId) == False:
            G_Campaign.AddNode(DstNId)
        G_Campaign.AddEdge(SrcNId, DstNId)
        eid = G_Campaign.GetEId(SrcNId, DstNId)
        G_Campaign.AddIntAttrDatE(eid, int(row['TRANSACTION_AMT']), 'TRANSACTION_AMT')

    return G_Campaign

def getCampaign_folded(G,legislator_node):
    G_Campaign_folded = snap.TUNGraph.New()

    for i in range(len(legislator_node)):
        for j in range(i+1,len(legislator_node)):
            Nbrs = snap.TIntV()
            if snap.GetCmnNbrs(G,legislator_node[i],legislator_node[j]) != 0:
                if G_Campaign_folded.IsNode(legislator_node[i]) == False:
                    G_Campaign_folded.AddNode(legislator_node[i])
                if G_Campaign_folded.IsNode(legislator_node[j]) == False:
                    G_Campaign_folded.AddNode(legislator_node[j])
                if G_Campaign_folded.IsEdge(legislator_node[i],legislator_node[j]) == False:
                    G_Campaign_folded.AddEdge(legislator_node[i],legislator_node[j]) 
            else:
                if G_Campaign_folded.IsNode(legislator_node[i]) == False:
                    G_Campaign_folded.AddNode(legislator_node[i])
                if G_Campaign_folded.IsNode(legislator_node[j]) == False:
                    G_Campaign_folded.AddNode(legislator_node[j])
            
    return G_Campaign_folded

def getY(G_CoSponsor,legislator_node):
    '''
    return a 3D pd df (node i, node j, link_result)
    link result = 0 if no collaboration, link result = 1 if collaboration
    '''
    Y = []
    #print legislator_node
    '''
    for i in range(len(legislator_node)):
        for j in range(i+1,len(legislator_node)):
            if i != j:
                result = G_CoSponsor.IsEdge(legislator_node[i],legislator_node[j])
                if result:
                    Y.append((legislator_node[i],legislator_node[j],1))
                else:
                    Y.append((legislator_node[i],legislator_node[j],-1))
    '''
    for i in range(len(legislator_node)):
        for j in range(i+1,len(legislator_node)):
            if i != j:
                result = G_CoSponsor.IsEdge(legislator_node[i],legislator_node[j])
                if result:
                    Y.append((legislator_node[i],legislator_node[j],1))
                else:
                    Y.append((legislator_node[i],legislator_node[j],-1))

    labels = ['node_i', 'node_j', 'result']
    Y = pd.DataFrame.from_records(Y, columns=labels)

    #print Y
    return Y

def getNeighbors(node, G):
    '''
    return neighbors node ID from TEANET
    '''
    neighbors = []
    NId = G.GetNI(node)
    #[neighbors.append(Id) for Id in G.GetNI(node).GetOutEdges()]

    for e in range(NId.GetInDeg()):
        neighbors.append(NId.GetInNId(e))

    return neighbors

def getFeatures(G_CoSponsor, G_Campaign, bill_node, legislator_node, comm_node,legislator_node_from_campaign,G_Campaign_folded):
    '''
    return two pd: X, Y
    '''    
    Y = getY(G_CoSponsor,legislator_node)
    print "before dropping", Y.shape
    
    for l in legislator_node:
        if l not in legislator_node_from_campaign:
            legislator_node.remove(l)
    
    Y = getY(G_CoSponsor,legislator_node)

    print "after dropping", Y.shape

    X = Y[['node_i', 'node_j']]

    #list of features

    X['Degree_Diff'] = 0
    X['Union_of_Neighbors'] = 0.0
    X['CommNeighbors'] = 0.0
    X['Contribution_Sum'] = 0.0
    X['Contribution_Diff'] = 0.0
    X['Clustering_Coeff_Diff'] = 0.0
    X['Clustering_Coeff_Sum'] = 0.0
    X['Clustering_Coeff_Avg'] = 0.0
    X['Jaccard'] = 0.0
    X['Shortest_Dist'] = 0.0
    X['Deg_Centrality_Diff'] = 0.0

    def compute_attri(x):
        NId_i = int(x['node_i'])
        NId_j = int(x['node_j'])
        if G_Campaign_folded.IsNode(NId_i) and G_Campaign_folded.IsNode(NId_j):
            node_i_contribution_sum = 0.0
            node_j_contribution_sum = 0.0
            neighbors_i = []
            neighbors_j = []


            clustering_cf_i = snap.GetNodeClustCf(G_Campaign_folded, NId_i)
            clustering_cf_j = snap.GetNodeClustCf(G_Campaign_folded, NId_j)

            CommNeighbors = snap.GetCmnNbrs(G_Campaign,NId_i,NId_j)
            NeighborsUnion = float(len(list(set().union(getNeighbors(NId_i,G_Campaign),getNeighbors(NId_j,G_Campaign)))))

            Nbrs = snap.TIntV()
            snap.GetCmnNbrs(G_Campaign, NId_i,NId_j, Nbrs)
            for NId in Nbrs:
                eid_i = G_Campaign.GetEId(NId,NId_i)
                eid_j = G_Campaign.GetEId(NId,NId_j)
                neighbors_i.append(NId)
                neighbors_j.append(NId)
                node_i_contribution_sum += G_Campaign.GetIntAttrDatE(eid_i, 'TRANSACTION_AMT')              
                node_j_contribution_sum += G_Campaign.GetIntAttrDatE(eid_j, 'TRANSACTION_AMT')
                
            result = {
                'Degree_Diff':abs(G_Campaign.GetNI(NId_i).GetInDeg() - G_Campaign.GetNI(NId_j).GetInDeg()),
                'Union_of_Neighbors': NeighborsUnion,
                'CommNeighbors': CommNeighbors,
                'Clustering_Coeff_Diff': abs(clustering_cf_i-clustering_cf_j),
                'Clustering_Coeff_Sum': clustering_cf_i+clustering_cf_j,
                'Clustering_Coeff_Avg': clustering_cf_i+clustering_cf_j/2.0,
                'Contribution_Diff': abs(node_i_contribution_sum - node_j_contribution_sum),
                'Contribution_Sum': node_i_contribution_sum + node_j_contribution_sum,
                'Jaccard': CommNeighbors*1.0/NeighborsUnion,
                'Shortest_Dist': snap.GetShortPath(G_Campaign, NId_i, NId_j),
                'Deg_Centrality_Diff': abs(snap.GetDegreeCentr(G_Campaign_folded,NId_i) - snap.GetDegreeCentr(G_Campaign_folded,NId_j))
            }
        else:
            result = {}
        return pd.Series(result,name="Attri")

    X = X.apply(compute_attri, axis = 1)
    print X
    print "before dropping nan", X.shape
    inds = pd.isnull(X).any(1).nonzero()[0]

    X =  X.drop(inds)
    Y =  Y.drop(inds)

    print "after dropping nan", X.shape

    return X, Y

def getlogistic(X, Y):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, Y)

    clf.predict(X)
    print clf.score(X, Y)
    return clf

def getSVC(X,Y):

    clf = SVC(gamma = 'auto')
    clf.fit(X, Y)

    clf.predict(X)
    print clf.score (X,Y)
    return clf


def getTree(X,Y):
    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X, Y)
    print clf.score(X,Y)
    return clf

def getBaselineFeatures(X, Y, party_df):
    X_base = X[['node_i', 'node_j']]
    print X.shape
    X_base = X_base.merge(party_df,left_on = 'node_i', right_on = "NodeID" ,how="left",)
    X_base = X_base.merge(party_df,left_on = 'node_j', right_on = "NodeID" ,how="left",)
    #X_base.to_csv('X_base.csv')
    return X, Y

def plotGridSearch(scores,xlabel,ylabel,title,axis):
    plt.plot(scores[:,0],scores[:,1])
    plt.plot(scores[:,0],scores[:,2])
    plt.legend(['training accuracy','testing accuracy'])
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=16)
    plt.axis(axis)    
    plt.show()
    pass

def main():
    '''
    Main script for link prediction:
        1) a function that takes bill sponsorship data from one term of Congresss and returns a vector of Y
    '''
    '''
    start_time = time.time()

    df = loadBillData(100) #get bill data for 100th congress
    #df = df.head(100)
    fin_df = loadFinancialData(1985,1986) #get financial data from two years prior
    #fin_df = fin_df.head(100)
    

    
    bill_node = df['SrcNId'].unique().tolist()
    legislator_node = df['DstNId'].unique().tolist()
    comm_node = fin_df['SrcNId'].unique().tolist()

    G_Campaign = getCampaign(fin_df)

    legislator_node_from_campaign = fin_df['DstNId'].unique().tolist()
    G_Campaign_folded = getCampaign_folded(G_Campaign,legislator_node_from_campaign)
        
    G_CoSponsor = getSponsorLink(df)
    
    snap.SaveEdgeList(G_Campaign, 'G_Campaign.txt')
    snap.SaveEdgeList(G_Campaign_folded, 'G_Campaign_folded.txt')
    snap.SaveEdgeList(G_CoSponsor, 'G_CoSponsor.txt')

    
    print "Get training"
    X, Y = getFeatures(G_CoSponsor,G_Campaign,bill_node, legislator_node, comm_node,legislator_node_from_campaign,G_Campaign_folded)
    
    X.to_csv('X.csv', index = False)
    Y.to_csv('Y.csv', index = False)
    
    print "My program took", time.time() - start_time, "to run train"

    print "Get test"
    #---get test set ---#
    df = loadBillData(101) #get bill data for 101th congress
    fin_df = loadFinancialData(1987,1988) #get financial data from two years prior
    
    
    bill_node = df['SrcNId'].unique().tolist()
    legislator_node = df['DstNId'].unique().tolist()
    comm_node = fin_df['SrcNId'].unique().tolist()

    G_Campaign = getCampaign(fin_df)

    legislator_node_from_campaign = fin_df['DstNId'].unique().tolist()
    G_Campaign_folded = getCampaign_folded(G_Campaign,legislator_node_from_campaign)
        
    G_CoSponsor = getSponsorLink(df)
    
    snap.SaveEdgeList(G_Campaign, 'G_Campaign.txt')
    snap.SaveEdgeList(G_Campaign_folded, 'G_Campaign_folded.txt')
    snap.SaveEdgeList(G_CoSponsor, 'G_CoSponsor.txt')

    X_test, Y_test = getFeatures(G_CoSponsor,G_Campaign,bill_node, legislator_node, comm_node,legislator_node_from_campaign,G_Campaign_folded)

    X_test.to_csv('X_test.csv', index = False)
    Y_test.to_csv('Y_test.csv', index = False)
    
    print "My program took", time.time() - start_time, "to run test"
    '''
    print "-----BEGAN CLASSIFICATION-----"   
    X = pd.read_csv('X.csv')
    Y = pd.read_csv('Y.csv')

    X_test = pd.read_csv('X_test.csv')
    Y_test = pd.read_csv('Y_test.csv')
    
    #------GET BASELINE WITH PARTYLINE INFORMATION ONLY------#
    party_df = loadParty('processed-data/candidate_node_mapping_manual_party.csv')
    #getBaselineFeatures(X,Y,party_df)

    print "baseline", Y[Y['result'] == 1].shape[0]/float(Y.shape[0])
    Y = Y['result']
    Y_test = Y_test['result']

    print "logistic"
    clf = getlogistic(X,Y)
    print clf.score(X_test,Y_test)

    selector = SelectPercentile(f_classif, percentile = 10)
    selector.fit(X, Y)

    Fval = []
    [Fval.append((list(X)[i],selector.scores_[i],selector.pvalues_[i])) for i in range(len(list(X)))]

    for i in Fval:
        print i

    #grid search for selectpercentile#
    scores = []

    for perc in range(5,80,5):
        selector = SelectPercentile(f_classif, percentile = perc)
        selector.fit(X, Y)
        print "using selector with percentile ", perc
        clf = getlogistic(selector.transform(X),Y)
        print clf.score(selector.transform(X_test),Y_test)
        scores.append((perc, clf.score(selector.transform(X),Y),clf.score(selector.transform(X_test),Y_test)))

    print scores
    scores = np.array(scores)

    plotGridSearch(scores,'selector percentile','accuracy','Grid search for optimal selector percentile',[0,80,0.5,1])

    print "tree"
    clf = getTree(X,Y)
    print clf.score(X_test,Y_test)
    '''
    #grid search for tree
    scores = []
    for depth in range(1,50,5):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X, Y)
        print clf.score(X,Y)
        print clf.score(X_test,Y_test)
        scores.append((depth, clf.score(X,Y),clf.score(X_test,Y_test)))
    
    scores = np.array(scores)
    plotGridSearch(scores,'max depth for decision tree classifier','accuracy','Grid search for optimal max depth for decision tree classifier',[0,50,0.5,1])


    '''
    '''
    print "My program took", time.time() - start_time, "to run score"
    '''
    '''
    print "SVC"
    clf_svc = getSVC(X,Y)
    print clf_svc.score(X_test,Y_test)
    '''
    
    pass


if __name__ == "__main__":
    main()