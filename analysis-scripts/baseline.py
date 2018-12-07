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
    print df
    return df

def getAttrBaseline(term,G_CoSponsor,legislator_node,legislator_node_from_campaign):
    cand_df = getCandidateAttri(term)
    accuracy = 0

    Y = link_prediction.getY(G_CoSponsor,legislator_node,legislator_node_from_campaign)  

    print list(Y)

    return accuracy


def main():

    pass

if __name__ == "__main__":
    main()