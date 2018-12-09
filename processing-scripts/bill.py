import snap
import pandas as pd
import glob

'''
This script imports data from govtrack_cosponsor_data and returns:
    - a bill_df.csv - a raw dataframe containing a concatenanted version of data
    - a legislator_node.csv - a list containing all legislators using their thomas_id & bioguide_id with node attributes (state)
    - a bill_node.csv - a list containing all bill_number with node attributes
    - a legislator_bill_edge_list - a list containing all edges between legislator and bill with edge attributes (congress term, date_signed), if withdraw, we do not add an edge
    - a snap edge list exported as txt containing candidate-bill projection with edge attribute (date_signed)
'''

pd.set_option('display.max_columns', None)

global input_path
global output_path

#path to the folder containing all govtrack_cosponsor_data, modify if needed
input_path = '/Users/yizhong/Documents/cs224w-data/bill data/govtrack_cosponsor_data'
output_path = '/Users/yizhong/Documents/cs224w-data/bill data/processed'

def loadData(path):
    '''
    This function takes a path to the folder containing all govtrack_cosponsor_data
    and returns a concatenated pd containing data from all congress
    '''
    allFiles = glob.glob(path + "/*.csv")
    df = pd.concat((pd.read_csv(f, delimiter= ",", parse_dates=True, keep_date_col=True, dtype = {'thomas_id': str, 'district': str, 'bioguide_id': str}, low_memory=False) for f in allFiles))

    df['thomas_id'] = df['thomas_id'].fillna('')
    df['bioguide_id'] = df['bioguide_id'].fillna('')
    df['combined_id'] = df['thomas_id'] + df['bioguide_id']
    df['congress_term'] = df['bill_number'].str.split("-").str[1]

    return df

def getLegislatorNode(df):
    legislator_node_t_id = df.groupby(['name',"thomas_id","state"]).size().reset_index(name='Freq')
    legislator_node_t_id = legislator_node_t_id[legislator_node_t_id['thomas_id'] != ""]

    legislator_node_bioguide_id = df.groupby(['name',"bioguide_id","state"]).size().reset_index(name='Freq')
    legislator_node_bioguide_id = legislator_node_bioguide_id[legislator_node_bioguide_id['bioguide_id'] != ""]

    legislator_node = legislator_node_bioguide_id.merge(legislator_node_t_id, left_on = ["name",'state'], right_on =["name",'state'],how = "outer")

    #NId for legislator starts from 0
    legislator_node["NId"] = legislator_node.reset_index().index
    
    print legislator_node.shape
    
    return legislator_node

def getBillNode(df):
    bill_node = df.groupby(['bill_number']).size().reset_index(name='Freq')

    #NId for node starts at 100000
    bill_node["NId"] = bill_node.reset_index().index
    bill_node["NId"] = bill_node["NId"] + 10000
    return bill_node

def getLegislatorBillEdges(df, legislator_node, bill_node):
    '''
    SrcNId: bill node #
    DstNId: legislator node #
    '''
    #remove all withdrawn cases
    df = df[df["date_withdrawn"].isnull()]

    legislator_bill_edge_list = df.groupby(['bill_number','state','name','congress_term','date_signed']).size().reset_index(name='Freq')

    #DstNId -> legislator, SrcNId -> bill_number
    legislator_bill_edge_list = legislator_bill_edge_list.merge(legislator_node, left_on=["name",'state'], right_on = ["name",'state'], how = "left")
    legislator_bill_edge_list = legislator_bill_edge_list.rename(index=str, columns={"NId": "DstNId"})

    legislator_bill_edge_list = legislator_bill_edge_list.merge(bill_node, left_on="bill_number", right_on = "bill_number", how = "left")
    legislator_bill_edge_list = legislator_bill_edge_list.rename(index=str, columns={"NId": "SrcNId"})

    print legislator_bill_edge_list[legislator_bill_edge_list['DstNId'].isnull()].shape

    return legislator_bill_edge_list[['SrcNId','DstNId','congress_term','date_signed','bill_number','name','state']]


def getGraph(legislator_bill_edge_list):
    G = snap.TUNGraph.New()
    df = legislator_bill_edge_list[['SrcNId','DstNId']]


    for index, row in df.iterrows():
        SrcNId = int(row['SrcNId'])
        DstNId = int(row['DstNId'])
        if G.IsNode(SrcNId) == False:
            G.AddNode(SrcNId)
        if G.IsNode(DstNId) == False:
            G.AddNode(DstNId)
        G.AddEdge(SrcNId, DstNId)
    
    print "number of nodes", G.GetNodes()
    print "number of edges", G.GetEdges()

    return G

if __name__ == "__main__":
    df = loadData(input_path)

    #return legislator_node.csv
    legislator_node = getLegislatorNode(df)
    legislator_node.to_csv(output_path + "/legislator_node.csv", index = False)
    
    #return bill_node.csv
    bill_node = getBillNode(df)
    bill_node.to_csv(output_path + "/bill_node.csv", index = False)

    #return legislator_bill_edge_list.csv
    legislator_bill_edge_list = getLegislatorBillEdges(df,legislator_node, bill_node)
    legislator_bill_edge_list.to_csv(output_path + "/legislator_bill_edge_list.csv", index = False)

    G= getGraph(legislator_bill_edge_list)
    snap.SaveEdgeList(G,output_path + "/legislator_bill_edge_list_graph.txt")
    