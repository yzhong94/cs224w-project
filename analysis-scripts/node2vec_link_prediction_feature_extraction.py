import link_prediction
import snap
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

def learn_embeddings(walks):
    '''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
    dimensions = 128
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=10, min_count=0, sg=1, workers=8, iter=3)
    model.wv.save_word2vec_format('embedding.emb')

def getEmbeddings(filename, p, q,walk_length):
    num_walks = 10

    nx_G = nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    for edge in nx_G.edges():
    	nx_G[edge[0]][edge[1]]['weight'] = 1
    

    G = node2vec.Graph(nx_G, False, p, q)
    
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks)


    pass

def getFeatures(G_CoSponsor, G_Campaign, bill_node, legislator_node, comm_node,legislator_node_from_campaign,emb):
    '''
    return two pd: X, Y based on embedding
    using: concatenate, Hadamard, Sum/Avg, Distance
    '''    
    #number of element-wise operations performed to emb
    total_num_operations = 3
    num_dimensions = emb.shape[1] - 1 #from embedding

    node_id = emb[:,0]
    
    for l in legislator_node:
        if l not in node_id:
            legislator_node.remove(l)
        if l not in legislator_node_from_campaign:
            try:
                legislator_node.remove(l)
            except:
                pass
    
    Y = link_prediction.getY(G_CoSponsor,legislator_node,legislator_node_from_campaign)

    print "after dropping", Y.shape
    num_operation = 0
    X = Y[['node_i', 'node_j']]
    for i in range(num_dimensions*total_num_operations):
        X[str(i)] = 0.0

    X['distance'] = 0.0
    print X.head(1)


    def compute_attri(x): 
        try:
            emb_i = emb[np.where(node_id==x['node_i']),1:][0][0]
            emb_j = emb[np.where(node_id==x['node_j']),1:][0][0]
            emb_sum = emb_i+emb_j
            emb_avg = np.mean([emb_i, emb_j],axis = 0)
            emb_hada = np.multiply(emb_i,emb_j)
            emb_dis = np.linalg.norm(emb_i - emb_j)
            #X['distance'][index] = emb_dis
            result = {
                "distance": emb_dis
            }
            #for i in range(num_dimensions):
            result.update({str(i):emb_sum[i] for i in range(num_dimensions)})
            result.update({str(i+num_dimensions):emb_avg[i] for i in range(num_dimensions)})
            result.update({str(i+num_dimensions*2):emb_hada[i] for i in range(num_dimensions)})
        except:
            result = {}
            pass
        

        return pd.Series(result,name="Attri")
    
    #X.head(50)
    X = X.apply(compute_attri, axis = 1)

    return X, Y

def getXYFromEmb(bill_term,fin_start_year, fin_end_year,p,q):
    df = link_prediction.loadBillData(bill_term) 
    fin_df = link_prediction.loadFinancialData(fin_start_year,fin_end_year) #get financial data from two years prior
    
    bill_node = df['SrcNId'].unique().tolist()
    legislator_node = df['DstNId'].unique().tolist()
    comm_node = fin_df['SrcNId'].unique().tolist()

    legislator_node_from_campaign = fin_df['DstNId'].unique().tolist()

    G_CoSponsor = link_prediction.getSponsorLink(df)

    G_Campaign = link_prediction.getCampaign(fin_df)

    snap.SaveEdgeList(G_Campaign, "G_campaign.txt")
    #G_Campaign = snap.LoadEdgeList(snap.PUNGraph, "G_campaign.txt", 0, 1)
    
    walk_length = 80
    getEmbeddings("G_campaign.txt", p, q,walk_length)
    

    print G_Campaign.GetNodes()
    
    emb = np.loadtxt('embedding.emb', skiprows = 1)
    node_id = emb[:,0]

    X, Y = getFeatures(G_CoSponsor, G_Campaign, bill_node, legislator_node, comm_node,legislator_node_from_campaign,emb)

    inds = pd.isnull(X).any(1).nonzero()[0]

    X =  X.drop(inds)
    Y =  Y.drop(inds)

    return X, Y

def main():
    '''
    returns embedding for 100th congress using p = 1, q = 0.5

    '''
    getXYFromEmb(100,1985, 1986, 1, 0.5)
    return 


if __name__ == "__main__":
    main()