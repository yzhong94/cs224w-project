import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random
import glob

CAM_YEAR = 1982
TERM_YEAR = 1984
TERM_START = 98 #1983-1984


filePath = "../processed-data/candidate_node_mapping_manual.csv"
mapping = pd.read_csv(filePath, index_col=False)

df = pd.DataFrame()

while CAM_YEAR <= 2016:


    filePath = "../data/financials/candidate_master/"
    candFile = filePath + str(CAM_YEAR) + ".txt"

    cand = pd.read_csv(candFile, sep='|', index_col=False, 
                     names=['CAND_ID',
                     'CAND_NAME',
                     'CAND_PTY_AFFILIATION',
                     'CAND_ELECTION_YR',
                     'CAND_OFFICE_ST',
                     'state',
                     'CAND_OFFICE',
                     'CAND_OFFICE_DISTRICT',
                     'CAND_ICI',
                     'CAND_STATUS',
                     'CAND_PCC',
                     'CAND_ST1',
                     'CAND_ST2',
                     'CAND_CITY',
                     'CAND_ST'])
    cand = cand[['CAND_ID','CAND_NAME', 'CAND_PTY_AFFILIATION', 
                 'CAND_ELECTION_YR','CAND_OFFICE_ST','state','CAND_OFFICE','CAND_ST2']]

    cand['CAM_YEAR'] = CAM_YEAR

    c = pd.merge(cand, mapping, on='CAND_ID', how='left')

    c[['NodeID']] = c[['NodeID']].apply(pd.to_numeric, errors='coerce').fillna(-1).astype(np.int64)
       
    ## remove entries where Nid = 0, meaning unsuccessful candidates
    c = c[c.fname != '']
    c = c[c.NodeID != -1]
    print c.shape

    df = pd.concat([df, c])

    CAM_YEAR = CAM_YEAR + 2

df.to_csv("../processed-data/party_candidates_attributes.csv", index = False)