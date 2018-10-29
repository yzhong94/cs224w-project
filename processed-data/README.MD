This folder hosts the processed data in the format of snap graph binary file.

Files
- a legislator_node.csv - a list containing all legislators using their thomas_id & bioguide_id with node attributes (state)
- a bill_node.csv - a list containing all bill_number with node attributes
- a legislator_bill_edge_list - a list containing all edges between legislator and bill with edge attributes (congress term, date_signed), if withdraw, we do not add an edge
- a simple snap edge list exported as txt containing candidate-bill projection