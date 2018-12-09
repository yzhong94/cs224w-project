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
from sklearn.metrics import confusion_matrix
import baseline
import itertools

start_time = time.time()

def loadClusteringAttr():
    cluster_0 = pd.read_csv('../processed-data/cluster_0.csv')
    cluster_1 = pd.read_csv('../processed-data/cluster_1.csv')
    #print cluster_0.squeeze().tolist()


    #return cluster_0['NodeId'].tolist(), cluster_1['NodeId'].tolist()
    return cluster_0.squeeze().tolist(), cluster_1.squeeze().tolist()

def loadBillData(term):
    '''
    For a given term, load the bill data (bill - candidate) and a pd dataframe with 
    SrcNId: bill node #, DstNId: legislator node
    Path: 'processed-data/legislator_bill_edge_list.csv'
    '''
    legislator_bill = pd.read_csv('../processed-data/legislator_bill_edge_list.csv')

    term_legislator_bill = legislator_bill[legislator_bill['congress_term'] == term]
    return term_legislator_bill[['SrcNId','DstNId']]

def loadFinancialData(start_year, end_year):
    '''
    For a given year range, load the financial data (candidate - committee) and a pd dataframe with 
    SrcNId: comm node #, DstNId: legislator node
    Path:'processed-data/campaignNetworks_raw_v2.csv'
    '''
    #----Need to run ----#
    financial_data = pd.read_csv('../processed-data/campaignNetworks_raw_v2.csv')
    term_financial_data = financial_data[(financial_data['ContributionYear'] >= start_year) & (financial_data['ContributionYear'] <= end_year )]

    term_financial_data = term_financial_data.rename(index=str, columns={"NodeID": "DstNId", "ComNodeId": "SrcNId"})
    return term_financial_data[['DstNId','SrcNId','TRANSACTION_AMT']]

def getSponsorLink(df):
    '''
    For a given term's bill df, return the cosponsor projection in snap undirected graph
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
    '''
    For a campaign network and node id for legislators, return an one-mode projection of campaign graphs
    '''
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
    return a 3D df (node i, node j, link_result)
    link result = 0 if no collaboration, link result = 1 if collaboration
    returned df does not contain duplicate link (node i,j v. node j,i)
    '''
    Y = []
    #print legislator_node

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
    
    return Y

def getNeighbors(node, G):
    '''
    return neighbors node ID from TEANET
    '''
    neighbors = []
    NId = G.GetNI(node)

    for e in range(NId.GetInDeg()):
        neighbors.append(NId.GetInNId(e))

    return neighbors

def getFeatures(G_CoSponsor, G_Campaign, bill_node, legislator_node, comm_node,legislator_node_from_campaign,G_Campaign_folded):
    '''
    return two pd: X, Y
    '''    
    print "before dropping",len(legislator_node)
    for l in legislator_node:
        if not G_Campaign_folded.IsNode(l):
            legislator_node.remove(l)
        if l not in legislator_node_from_campaign:
            try:
                legislator_node.remove(l)
            except:
                pass
    
    cluster_0, cluster_1 = loadClusteringAttr()
    print "after dropping",len(legislator_node)
 
    Y = getY(G_CoSponsor,legislator_node)

    #compute a list of clustering coefficient
    NIdCCfH = snap.TIntFltH()
    snap.GetNodeClustCf(G_Campaign_folded, NIdCCfH)

    #compute a list of node centrality and degree
    node_centrality={}
    in_deg = {}
    for i in legislator_node:
        if G_Campaign.IsNode(i):
            node_centrality[i]=snap.GetDegreeCentr(G_Campaign_folded,i)
            in_deg[i] = G_Campaign.GetNI(i).GetInDeg()
    
    print "begin to compute X"

    X = Y[['node_i', 'node_j']]

    #list of features
    X['Degree_Diff'] = 0
    X['Union_of_Neighbors'] = 0.0
    X['CommNeighbors'] = 0.0
    #X['Contribution_Sum'] = 0.0
    #X['Contribution_Diff'] = 0.0
    X['Clustering_Coeff_Diff'] = 0.0
    X['Clustering_Coeff_Sum'] = 0.0
    X['Clustering_Coeff_Avg'] = 0.0
    X['Jaccard'] = 0.0
    X['Shortest_Dist'] = 0.0
    X['Deg_Centrality_Diff'] = 0.0
    X['FromSameCluster'] = 0


    def compute_attri(x):
        NId_i = int(x['node_i'])
        NId_j = int(x['node_j'])
        if G_Campaign_folded.IsNode(NId_i) and G_Campaign_folded.IsNode(NId_j):
            node_i_contribution_sum = 0.0
            node_j_contribution_sum = 0.0
            neighbors_i = []
            neighbors_j = []

            clustering_cf_i = NIdCCfH[NId_i]

            clustering_cf_j = NIdCCfH[NId_j]
        
            CommNeighbors = snap.GetCmnNbrs(G_Campaign,NId_i,NId_j)
            NeighborsUnion = float(len(list(set().union(getNeighbors(NId_i,G_Campaign),getNeighbors(NId_j,G_Campaign)))))

            FromSameCluster = 0
            if NId_i in cluster_0 and NId_j in cluster_0:
                FromSameCluster = 1
            if NId_i in cluster_1 and NId_j in cluster_1:
                FromSameCluster = 1
            '''
            Nbrs = snap.TIntV()
            snap.GetCmnNbrs(G_Campaign, NId_i,NId_j, Nbrs)
            for NId in Nbrs:
                eid_i = G_Campaign.GetEId(NId,NId_i)
                eid_j = G_Campaign.GetEId(NId,NId_j)
                neighbors_i.append(NId)
                neighbors_j.append(NId)
                node_i_contribution_sum += G_Campaign.GetIntAttrDatE(eid_i, 'TRANSACTION_AMT')              
                node_j_contribution_sum += G_Campaign.GetIntAttrDatE(eid_j, 'TRANSACTION_AMT')
            '''
            result = {
                'Degree_Diff':abs(in_deg[NId_i] - in_deg[NId_j]),
                'Union_of_Neighbors': NeighborsUnion,
                'CommNeighbors': CommNeighbors,
                'Clustering_Coeff_Diff': abs(clustering_cf_i-clustering_cf_j),
                'Clustering_Coeff_Sum': clustering_cf_i+clustering_cf_j,
                'Clustering_Coeff_Avg': clustering_cf_i+clustering_cf_j/2.0,
                #'Contribution_Diff': abs(node_i_contribution_sum - node_j_contribution_sum),
                #'Contribution_Sum': node_i_contribution_sum + node_j_contribution_sum,
                'Jaccard': CommNeighbors*1.0/NeighborsUnion,
                'Shortest_Dist': snap.GetShortPath(G_Campaign, NId_i, NId_j),
                'Deg_Centrality_Diff': abs(node_centrality[NId_i] - node_centrality[NId_j]),
                'FromSameCluster': FromSameCluster
            }
        else:
            result = {}
        return pd.Series(result,name="Attri")

    begin = time.time()
    print "My program took", time.time() - start_time, "to begin compute X"

    X = X.apply(compute_attri, axis = 1)
    print "before dropping nan from computing attribute", X.shape
    inds = pd.isnull(X).any(1).nonzero()[0]
    print "My program took", time.time() - start_time, "to finish compute X"
    end = time.time()

    print "time to compute x", begin - end

    X =  X.drop(inds)
    Y =  Y.drop(inds)

    print "after dropping nan from computing attribute", X.shape

    return X, Y

def getlogistic(X, Y):
    clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='ovr',max_iter = 10000000).fit(X, Y)

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

def getFeaturesForTerm(term):
    df = loadBillData(term) #get bill data for a specific term
    start_year, end_year = common_function.getTermMapping(term)
    fin_df = loadFinancialData(start_year-2 ,end_year-2) #get financial data from two years prior
    
    bill_node = df['SrcNId'].unique().tolist()
    legislator_node = df['DstNId'].unique().tolist()
    comm_node = fin_df['SrcNId'].unique().tolist()
    legislator_node_from_campaign = fin_df['DstNId'].unique().tolist()

    
    G_Campaign = getCampaign(fin_df)
    
    G_CoSponsor = getSponsorLink(df)
    G_Campaign_folded = getCampaign_folded(G_Campaign,legislator_node_from_campaign)
    
    snap.SaveEdgeList(G_Campaign, 'G_Campaign.txt')
    snap.SaveEdgeList(G_Campaign_folded, 'G_Campaign_folded.txt')
    snap.SaveEdgeList(G_CoSponsor, 'G_CoSponsor.txt')
    '''
    G_Campaign_folded = snap.LoadEdgeList(snap.PUNGraph, 'G_Campaign_folded.txt',0,1)
    G_CoSponsor = snap.LoadEdgeList(snap.PUNGraph, 'G_CoSponsor.txt',0,1)
    '''
    X, Y = getFeatures(G_CoSponsor,G_Campaign,bill_node, legislator_node, comm_node,legislator_node_from_campaign,G_Campaign_folded) 
    X['term'] = term
    return X, Y

def plot_confusion_matrix(clf, X_test, Y_test, title):

    pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test,pred)
    print(cm)
    plt.figure()
    plt.title('Normalized Confusion Matrix for ' + title)
    fmt = '.2f'
    classes = [-1,1]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.matshow(cm)
    plt.show()

    pass

def main():
    '''
    Main script for link prediction
    '''
    '''
    print "-----LIMITED SAMPLES-----"
    print "-----BEGIN EXTRACTING FEATURES-----"
    
    term = 100
    '''
    '''
    X,Y = getFeaturesForTerm(term)

    X.to_csv('X.csv', index = False)
    Y.to_csv('Y.csv', index = False)
    
    print "My program took", time.time() - start_time, "to run train"
    
    X_test,Y_test = getFeaturesForTerm(term+1)
    X_test.to_csv('X_test.csv', index = False)
    Y_test.to_csv('Y_test.csv', index = False)
    
    print "My program took", time.time() - start_time, "to run test"
    '''
    
    
    print "-----ALL SAMPLES: 98th-112th-----"
    print "-----BEGIN EXTRACTING FEATURES-----"
    
    X_all,Y_all = getFeaturesForTerm(98)

    for term in range(99,112,1):
        print "looping term", term
        X,Y = getFeaturesForTerm(term)
        X_all = pd.concat([X_all,X],ignore_index=True)
        Y_all = pd.concat([Y_all,Y],ignore_index=True)
        X_all.to_csv('X_all.csv', index = False)
        Y_all.to_csv('Y_all.csv', index = False)
    

    print "My program took", time.time() - start_time, "to run train"
    
    X_test_113,Y_test_113 = getFeaturesForTerm(113)
    X_test_114,Y_test_114 = getFeaturesForTerm(114)

    X_test_all = pd.concat([X_test_113,X_test_114],ignore_index=True)
    Y_test_all = pd.concat([Y_test_113,Y_test_114],ignore_index=True)

    X_test_all.to_csv('X_test_all.csv', index = False)
    Y_test_all.to_csv('Y_test_all.csv', index = False)
    
    print "My program took", time.time() - start_time, "to run test"

    
    '''
    print "-----BEGAN CLASSIFICATION-----" 
    print "-----LIMITED SAMPLES-----" 
    
    X = pd.read_csv('X.csv')
    Y = pd.read_csv('Y.csv')

    X_test = pd.read_csv('X_test.csv')
    Y_test = pd.read_csv('Y_test.csv')
    '''
    '''
    print "-----ALL SAMPLES-----"    
    X = pd.read_csv('X_all.csv')
    Y = pd.read_csv('Y_all.csv')

    X_test = pd.read_csv('X_test_all.csv')
    Y_test = pd.read_csv('Y_test_all.csv')
    #Y_test = Y_test[X_test['term']==113]
    #X_test = X_test[X_test['term']==113]
    '''

    
    #------GET BASELINE WITH PARTYLINE INFORMATION ONLY------#
    print "My program took", time.time() - start_time, "to begin baseline"

    X_base_train, Y_base_train = baseline.getAttrBaseline(term,Y)
    X_base_test, Y_base_test = baseline.getAttrBaseline(term+1,Y_test)
    #print X_base_test.head()
    #print Y_base_test.head()

    print "My program took", time.time() - start_time, "to end baseline"

    print "logistic baseline"
    clf = getlogistic(X_base_train,Y_base_train)
    print clf.score(X_base_test,Y_base_test)
    plot_confusion_matrix(clf, X_test, Y_test, 'Logistic Baseline - Candidate Info Only')
    
    print "naive baseline", Y[Y['result'] == 1].shape[0]/float(Y.shape[0])
    
    
    Y = Y['result']
    Y_test = Y_test['result']
    
    print "logistic"
    clf = getlogistic(X,Y)
    print clf.score(X_test,Y_test)

    plot_confusion_matrix(clf, X_test, Y_test, 'Logistic Regression')

    
    '''
    selector = SelectPercentile(f_classif, percentile = 10)
    selector.fit(X, Y)

    Fval = []
    [Fval.append((list(X)[i],selector.scores_[i],selector.pvalues_[i])) for i in range(len(list(X)))]

    for i in Fval:
        print i
    '''

    '''
    #grid search for selectpercentile#
    scores = []

    for perc in range(5,100,5):
        selector = SelectPercentile(f_classif, percentile = perc)
        selector.fit(X, Y)
        print "using selector with percentile ", perc
        clf = getlogistic(selector.transform(X),Y)
        print clf.score(selector.transform(X_test),Y_test)
        scores.append((perc, clf.score(selector.transform(X),Y),clf.score(selector.transform(X_test),Y_test)))

    scores = np.array(scores)
    
    plotGridSearch(scores,'selector percentile','accuracy','Grid search for optimal selector percentile',[0,80,0.5,1])
    '''
    
    print "tree"
    clf = getTree(X,Y)
    print clf.score(X_test,Y_test)

    plot_confusion_matrix(clf, X_test, Y_test, 'Decision Tree')

    '''
    #grid search for tree
    scores = []
    for depth in range(1,50,5):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X, Y)
        print depth
        print clf.score(X,Y)
        print clf.score(X_test,Y_test)
        scores.append((depth, clf.score(X,Y),clf.score(X_test,Y_test)))
    
    scores = np.array(scores)
    plotGridSearch(scores,'max depth for decision tree classifier','accuracy','Grid search for optimal max depth for decision tree classifier',[0,50,0.5,1])

    '''
    
    print "My program took", time.time() - start_time, "to run score"
    '''
    print "SVC"
    clf_svc = getSVC(X,Y)
    print clf_svc.score(X_test,Y_test)
    '''
    
    pass


if __name__ == "__main__":
    main()