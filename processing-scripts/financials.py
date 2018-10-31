import snap
import numpy as np
import pandas as pd
from itertools import permutations
from matplotlib import pyplot as plt
import random


def loadCandidateMaster(filename):

	result = pd.read_csv(filename, sep='|', index_col=False, 
                 names=['CAND_ID',
                 'CAND_NAME',
                 'CAND_PTY_AFFILIATION',
                 'CAND_ELECTION_YR',
                 'CAND_OFFICE_ST',
                 'CAND_OFFICE',
                 'CAND_OFFICE_DISTRICT',
                 'CAND_ICI',
                 'CAND_STATUS',
                 'CAND_PCC',
                 'CAND_ST1',
                 'CAND_ST2',
                 'CAND_CITY',
                 'CAND_ST',
                 'CAND_ZIP'])

	return result

if __name__ == "__main__":

	cn2012 = loadCandidateMaster("../data/financials/candidate_master/2012.txt")
	print(f.head())