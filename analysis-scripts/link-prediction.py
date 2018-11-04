import snap
import pandas as pd



def loadBillData(term):
    '''
    For a given term, load the bill data (bill - candidate) and a pd dataframe with 
    SrcNId: bill node #, DstNId: legislator node
    '''
    legislator_bill = pd.read_csv('processed-data/legislator_bill_edge_list.csv')

    term_legislator_bill = legislator_bill[legislator_bill['congress_term'] == term]
    return term_legislator_bill[[SrcNId,DstNId]]

def loadFinancialData(year):
    '''
    For a given year, load the bill data (bill - candidate) and a pd dataframe with 
    SrcNId: bill node #, DstNId: legislator node
    '''

    return

def main():
    '''
    Main script for link prediction:
        1) a function that takes bill sponsorship data from one term of Congresss and returns a vector of Y
    '''
    pd = loadBillData(100)

    pass


if __name__ == "__main__":
    main()