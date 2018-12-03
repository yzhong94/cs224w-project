import snap
import pandas as pd
import bill
import common_function
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
import networkx

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

def loadParty(df):
    party_data = pd.read_csv('')

    return

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
    For a given term's campaign data, return a graph with edge attributes of contribution amount
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

def getY(G_CoSponsor,legislator_node):
    '''
    return a 3D pd df (node i, node j, link_result)
    link result = 0 if no collaboration, link result = 1 if collaboration
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
    #[neighbors.append(Id) for Id in G.GetNI(node).GetOutEdges()]

    for e in range(NId.GetInDeg()):
        neighbors.append(NId.GetInNId(e))

    return neighbors

def getFeatures(G_CoSponsor, G_Campaign, bill_node, legislator_node, comm_node,legislator_node_from_campaign):
    '''
    return two pd: X, Y
    '''    
    Y = getY(G_CoSponsor,legislator_node)
    print "before dropping", Y.shape
    
    for l in legislator_node:
        if l not in legislator_node_from_campaign:
            legislator_node.remove(l)

    print "after dropping", len(legislator_node)
    
    Y = getY(G_CoSponsor,legislator_node)

    print "after dropping", Y.shape

    #first feature: number of common neighbors between node i and node j

    X = Y[['node_i', 'node_j']]

    X['Degree_Diff'] = 0
    X['Union_of_Neighbors'] = 0
    X['CommNeighbors'] = 0
    X['Contribution_Sum'] = 0
    X['Contribution_Diff'] = 0
    #X['CommNeighbors'] = X.apply(lambda row: snap.GetCmnNbrs(G_Campaign,row['node_i'],row['node_j']),axis = 1)

    for index, row in X.iterrows():
        node_i_contribution_sum = 0
        node_j_contribution_sum = 0
        neighbors_i = []
        neighbors_j = []
        
        if G_Campaign.IsNode(row['node_i']) == False or G_Campaign.IsNode(row['node_j']) == False:
            X.drop(X.index[index])
            Y.drop(Y.index[index])
        else:
            X['Degree_Diff'][index] = abs(G_Campaign.GetNI(row['node_i']).GetInDeg() - G_Campaign.GetNI(row['node_j']).GetInDeg())
            X['Union_of_Neighbors'][index] = float(len(list(set().union(getNeighbors(row['node_i'],G_Campaign),getNeighbors(row['node_j'],G_Campaign)))))
            #print "Union_of_Neighbors", X['Union_of_Neighbors'][index]
            X['CommNeighbors'][index] = snap.GetCmnNbrs(G_Campaign,row['node_i'],row['node_j'])
            #print "CommNeighbors", X['CommNeighbors'][index]
            
            Nbrs = snap.TIntV()
            snap.GetCmnNbrs(G_Campaign, row['node_i'],row['node_j'], Nbrs)
            #print row['node_i'], row['node_j']
            for NId in Nbrs:
                eid_i = G_Campaign.GetEId(NId,row['node_i'])
                eid_j = G_Campaign.GetEId(NId,row['node_j'])
                neighbors_i.append(NId)
                neighbors_j.append(NId)
                node_i_contribution_sum += G_Campaign.GetIntAttrDatE(eid_i, 'TRANSACTION_AMT')              
                node_j_contribution_sum += G_Campaign.GetIntAttrDatE(eid_j, 'TRANSACTION_AMT')
            
            X['Contribution_Sum'][index] = node_i_contribution_sum + node_j_contribution_sum
            X['Contribution_Diff'][index] = abs(node_i_contribution_sum - node_j_contribution_sum)

    X['Jaccard'] = X['CommNeighbors'].astype(float)/X['Union_of_Neighbors'].astype(float)
    
    #print X['Jaccard']
    X['Jaccard'].replace(np.inf,0.0)
    X['Jaccard'] = X['Jaccard'].fillna(0.0)

    print "-----    DONE    -----"


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

def main():
    '''
    Main script for link prediction:
        1) a function that takes bill sponsorship data from one term of Congresss and returns a vector of Y
    '''
    '''
    
    financial_data = pd.read_csv('processed-data/campaignNetworks_raw.csv')
    bill_data = pd.read_csv('processed-data/legislator_bill_edge_list.csv')
    financial_data = financial_data.rename(index=str, columns={"NodeID": "DstNId", "ComNodeId": "SrcNId"})


    
    G = common_function.getGraph(financial_data)
    
    legislator_node = bill_data['DstNId'].unique()

    non_node = []
    for i in legislator_node:
        if G.IsNode(i) == False:
            non_node.append(i)

    print "total nodes", len(legislator_node)
    print len(non_node)
    '''
    
    
    
    df = loadBillData(100) #get bill data for 100th congress
    #df = df.head(100)
    fin_df = loadFinancialData(1985,1986) #get financial data from two years prior
    #fin_df = fin_df.head(100)

    bill_node = df['SrcNId'].unique().tolist()
    legislator_node = df['DstNId'].unique().tolist()
    comm_node = fin_df['SrcNId'].unique().tolist()

    legislator_node_from_campaign = fin_df['DstNId'].unique().tolist()

    G_CoSponsor = getSponsorLink(df)

    G_Campaign = getCampaign(fin_df)

    X, Y = getFeatures(G_CoSponsor,G_Campaign,bill_node, legislator_node, comm_node,legislator_node_from_campaign)
    
    X.to_csv('X.csv', index = False)
    Y.to_csv('Y.csv', index = False)
    
    
    #---get test set ---#
    df = loadBillData(101) #get bill data for 101th congress
    fin_df = loadFinancialData(1987,1988) #get financial data from two years prior
    
    
    bill_node = df['SrcNId'].unique().tolist()
    legislator_node = df['DstNId'].unique().tolist()
    comm_node = fin_df['SrcNId'].unique().tolist()
    legislator_node_from_campaign = fin_df['DstNId'].unique().tolist()

    G_CoSponsor = getSponsorLink(df)

    G_Campaign = getCampaign(fin_df)


    X_test, Y_test = getFeatures(G_CoSponsor,G_Campaign,bill_node, legislator_node, comm_node,legislator_node_from_campaign)

    X_test.to_csv('X_test.csv', index = False)
    Y_test.to_csv('Y_test.csv', index = False)
    
    
    print "-----BEGAN CLASSIFICATION-----"    
    X = pd.read_csv('X.csv')
    Y = pd.read_csv('Y.csv')

    X_test = pd.read_csv('X_test.csv')
    Y_test = getY(G_CoSponsor,legislator_node)

    X = X.drop(columns=['node_i','node_j'])
    X_test = X_test.drop(columns=['node_i','node_j'])

    print "baseline", Y[Y['result'] == 1].shape[0]/float(Y.shape[0])
    Y = Y['result']
    Y_test = Y_test['result']

    print "logistic"
    clf = getlogistic(X,Y)
    print clf.score(X_test,Y_test)

    print "tree"
    clf = getTree(X,Y)
    print clf.score(X_test,Y_test)
    
    '''
    print "SVC"
    clf_svc = getSVC(X,Y)
    print clf_svc.score(X_test,Y_test)
    '''
    pass


if __name__ == "__main__":
    main()