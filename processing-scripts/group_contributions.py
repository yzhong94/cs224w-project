import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("../processed-data/campaignNetworks_raw_v2.csv", index_col = False)

df = df.groupby(['CMTE_ID', 'ContributionYear', 'NodeID'], as_index=False).agg({"TRANSACTION_AMT": "sum"})

print df.shape