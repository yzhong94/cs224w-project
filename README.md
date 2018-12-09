# cs224w-project
Link Prediction in Congress Bill Co-Sponsorship Networks Using Political Donor Network Information

# Processed Data Directory
- "../processed-data/party_candidates_attributes.csv": this contains party and state attributes for candidates. Checked for data consistency, and that each candidate only has one party affiliation in a given year
- "../processed-data/candidate_node_mapping_manual.csv": manually inspected mapping between CAND_ID (from campaign files and records) to NodeID, as NodeID is the unified NodeID we use throughout the project
- "../processed-data/campaignNetworks_v2.txt": Campaign Network, a bipartite graph between candidates and committees. An edge is formed when a committee donated to a candidate. This list is for all years combined (1981 - 2016)
- "../processed-data/legislator_bill_edge_list.csv": - a legislator_bill_edge_list - a list containing all edges between legislator and bill with edge attributes (congress term, date_signed).  If a legislator withdrawn from the sponsoring process, an edge is not added.
- "../processed-data/legislator_node.csv": a list containing all legislators using their thomas_id & bioguide_id with node attributes (state) from the bill network
- "../processed-data/bill_node.csv": a list containing all bill_number
- "../processed-data/legislator_bill_edge_list.csv"

# Script Directory
- "../processing-scripts/candidate_mapping.py": this generates campaign network edge list for all years of campaign records combined, for candidates we can find in the bills data (i.e. candidates who eventually got elected into office)
- "../processing-scripts/partyAttributes.py": Parse party and state data from candidate master, and map it to unified nodeIDs for predictions
- "../analysis-scripts/describe.py": Folds bi-partite graphs, describe degree distributions 
- "../analysis-scripts/spectralClustering.ipynb": iPython notebook to run spectral clustering based on Clauset-Newman-Moore greedy modularity maximization
- "../analysis-scripts/loopThruEachYear.py": This outputs every batch's network info (NOTE: not a combined, aggregate graph; but have a graph only using one term's data - e.g. 1981-1982's campaign records with 1983-1984's bill co-authorship data)
- "../analysis-scripts/link_prediction.py": the main script for all link prediction analysis, includes code for: 1) generating features, 2) computing baseline accuracy, 3) run machine learning algorithms for link prediction, 4) perform grid search for optimal parameters
- "../analysis-scripts/node2vec_link_prediction_analysis.py": the script for learning embeddings from node2vec walks and running logistic regression