import snap
import numpy as np
import matplotlib.pyplot as plt

def readGraph(path):
    G = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1)
    return G

## This function takes in a graph, and computes cosine similarity between nodes and the graph's median node
## Reusing the code from hw2 q1
def basicFeature(G):
	V = []
	cnt = 0
	x1 = 0
	x2 = 0
	x3 = 0
	for NI in G.Nodes():
		## Get egonet
		NIdV = snap.TIntV()
		NIdV.Add(NI.GetId())
		for Id in NI.GetOutEdges():
			NIdV.Add(Id)
		results = snap.GetEdgesInOut(G, NIdV)
		V.append([NI.GetId(), NI.GetOutDeg(),results[0], results[1]])
		cnt = cnt + 1
		x1 = x1 + NI.GetOutDeg()
		x2 = x2 + results[0]
		x3 = x3 + results[1]

	Id9 = 9999 ##hard code the biggest possible node value for candidates to avoid collision
	x1 = x1 * 1.0/cnt
	x2 = x2 * 1.0/cnt
	x3 = x3 * 1.0/cnt

	a = np.sqrt(x1 * x1 + x2 * x2 + x3* x3)
	res = []
	scores = []

	for i in V:
		[Id, y1, y2, y3] = i
		if Id != Id9:
			b = np.sqrt(y1 * y1 + y2 * y2 + y3* y3)
			dem = x1 * y1 + x2 * y2 + x3 * y3
			if (b == 0 or a == 0 or dem == 0):
				sim = 0
			else:
				sim = dem*1.0/a/b
			res.append([Id, sim])
			scores.append(sim)
	
	arr = np.array(res)
	r = arr[arr[:, 1].argsort()]
	l = len(res)
	print "Top 5 similar nodes based on cosine similarity, for basic features like HW2 Q1, when compared to mean node"
	print(r[l-1], r[l-2], r[l-3], r[l-4], r[l-5]) 

	###
	## print roles
	plt.figure()
	plt.hist(scores, bins=20)
	plt.title('Distribution of cosine similarity between mean node and any node in the graph')
	plt.xlabel('cosine similarity')
	plt.ylabel('count')
	plt.show()
	
	###

	return

def recursiveFeature(G, N):
	VId = []
	#print "node count is %d" % (G.GetNodes())
	for NI in G.Nodes():
		VId.append(NI.GetId())

	max_node = max(VId)

	V = []
	V9 = []

	for NId in VId:
		NI = G.GetNI(NId)
		NIdV = snap.TIntV()
		NIdV.Add(NId)
		for Id in NI.GetOutEdges():
			NIdV.Add(Id) # all the neighbors
		results = snap.GetEdgesInOut(G, NIdV)
		V.append([NI.GetOutDeg(),results[0], results[1]])
		if NI.GetId() == N:
			V9 = [NI.GetOutDeg(),results[0], results[1]]

	K = 2
	#print(V9)
	V = np.array(V)
	VId = np.array(VId)

	Vnew = []
	iter = 0
	while iter < K:
		# for each node
		for NI in range(len(V)):
			idx = VId[NI]
			NodeI = G.GetNI(idx)
			Nbr = []
			#get neighbors features
			Nbr = getNbrFeatures(G, idx, V, VId)			
			#append to NI feature vector
			if len(Nbr) == 0:
				Nbr_mean = np.zeros(V[NI].shape)
				Nbr_sum = np.zeros(V[NI].shape)
				#print(Nbr_mean, 'found zero neighbors')
			else:
				Nbr_mean = np.array(Nbr).mean(axis=0)
				Nbr_sum = np.array(Nbr).sum(axis=0)
			Vnew.append(np.concatenate((V[NI], Nbr_mean, Nbr_sum)))
			#if len(Nbr) == 0:
				#print(np.concatenate((V[NI], Nbr_mean, Nbr_sum)))

			if idx == N:
				V9 = np.concatenate((V[NI], Nbr_mean, Nbr_sum))
		iter = iter + 1
		V = np.array(Vnew)
		Vnew = []
	# now calculate similarity
	res = []

	#rint(V.shape)
	a = np.linalg.norm(V9)
	scores = []
	for NI in range(len(V)):
		if VId[NI] != N: 
			b = np.linalg.norm(V[NI])
			dem = np.dot(V9, V[NI])
			if (b == 0 or a == 0 or dem == 0):
				sim = 0
			else:
				sim = dem*1.0/a/b
			res.append([VId[NI], sim])
			scores.append(sim)

	arr = np.array(res)
	r = arr[arr[:, 1].argsort()]
	l = len(res)
	#print(l)
	print("Recursive Features: ")
	print(r[l-1], r[l-2], r[l-3], r[l-4], r[l-5]) 

	###
	
	plt.figure()
	plt.hist(scores, bins=20)
	plt.title('Distribution of cosine similarity between node 322 and any other node in the graph')
	plt.xlabel('cosine similarity')
	plt.ylabel('count')
	plt.show()
	
	###
	
	## print subgraphs for roles: 0.6, 0.87, 0.92 and 0.97
	a = np.linalg.norm(V9)
	for NI in range(len(V)):
		if VId[NI] != N: 
			b = np.linalg.norm(V[NI])
			dem = np.dot(V9, V[NI])
			if (b == 0 or a == 0 or dem == 0):
				sim = 0
				#print "FOUND ZERO!!!!!!"
			else:
				sim = dem*1.0/a/b
		if np.abs(sim-0.0) < 0.001:
			print(0, VId[NI], V[NI])
		if np.abs(sim-0.6) < 0.001:
			print(0.6, VId[NI], V[NI])
		if np.abs(sim-0.85) < 0.001:
			print(0.85, VId[NI], V[NI])
		if np.abs(sim-0.9) < 0.001:
			print(0.9, VId[NI], V[NI])
	
	return

def getNbrFeatures(G, NodeID, V, VId):
## given a node ID, feature vector, and a graph, return Node's neighbors' features
	NodeI = G.GetNI(NodeID)
	Nbr = []
	NbrID = []
	for Id in NodeI.GetOutEdges():
		NbrID.append(Id)
	for i in range(len(V)):
		if (VId[i] in NbrID):
			Nbr.append(V[i])

	#print('inside GetNbrs, ', Nbr)
	return Nbr

def role(Graph):
	## load folded graphs
    if Graph == 'campaign':
        G = readGraph("../processed-data/campaign_projection.txt")
    elif Graph == 'bill':
        G = readGraph("../processed-data/bill_projection.txt")
    else:
        raise ValueError("Invalid graph: please use 'campaign' or 'bill'. ")

    #basicFeature(G)
    recursiveFeature(G, 322)

    return

if __name__ == "__main__":
	
	#role("bill")
	role("campaign")
	#print "Done with Question 1.1!"

	#recursiveFeature(G, 9)

	print "Done with Role Exploration!"