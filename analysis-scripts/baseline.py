import link_prediction
import snap
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import common_function

def getCandidateAttri(term):
    df = pd.read_csv('processed-data/party_candidates_attributes.csv')
    start_year, end_year = common_function.getTermMapping(term)

    df = df[df['CAM_YEAR']>=start_year]
    df = df[df['CAM_YEAR']<=end_year]
    print list(df)
    df = df.groupby(['NodeID','state_y','CAND_PTY_AFFILIATION']).size().reset_index(name='Freq')
    #print df
    return df

def getAttrBaseline(term,Y):
    cand_df = getCandidateAttri(term)
    accuracy = 0

    X = Y[['node_i', 'node_j']]
    print list(Y)


    def compute_attri(x):
        fromSameParty=0
        fromSameState=0
        NId_i = int(x['node_i'])
        NId_j = int(x['node_j'])
        NId_i_party = cand_df[cand_df['NodeID']==NId_i]['CAND_PTY_AFFILIATION'].tolist()[0]
        NId_j_party = cand_df[cand_df['NodeID']==NId_j]['CAND_PTY_AFFILIATION'].tolist()[0]
        NId_i_state = cand_df[cand_df['NodeID']==NId_i]['state_y'].tolist()[0]
        NId_j_state = cand_df[cand_df['NodeID']==NId_j]['state_y'].tolist()[0]

        if NId_i_party == NId_j_party:
            fromSameParty = 1
        if NId_i_state == NId_j_state:
            fromSameState = 1

        result = {
            'fromSameParty':fromSameParty,
            'fromSameState':fromSameState
        }

        return pd.Series(result,name="Attri")
    '''
    X = X.apply(compute_attri, axis = 1)
    print X
    '''
    X = X.merge(cand_df,left_on = "node_i",right_on = "NodeID",how = "left")
    X = X.merge(cand_df,left_on = "node_j",right_on = "NodeID",how = "left")
    X['fromSameParty'] = ''
    X['fromSameState'] = ''
    print X[X['CAND_PTY_AFFILIATION_y'] == X['CAND_PTY_AFFILIATION_x']].shape
    
    X['fromSameParty'][X['CAND_PTY_AFFILIATION_y'] == X['CAND_PTY_AFFILIATION_x']] = 1
    X['fromSameParty'][X['CAND_PTY_AFFILIATION_y'] != X['CAND_PTY_AFFILIATION_x']] = 0

    X['fromSameState'][X['state_y_y'] == X['state_y_x']] = 1
    X['fromSameState'][X['state_y_y'] != X['state_y_x']] = 0

    X.to_csv('test.csv')

    return X, Y


def main():

    pass

if __name__ == "__main__":
    main()