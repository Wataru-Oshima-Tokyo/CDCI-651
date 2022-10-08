import numpy as np
import networkx as nx

def _mean_estimate(G):
    for p in np.arange(0.1,1.0,0.1):
        estimates ={}
        for _ in range(100):
            nodes = np.random.choice(G.nodes(), size=int(p*len(G)), replace=False)
            estimates[p] = estimates.get(p,[]) + [np.mean([G.degree(v) for v in nodes])]
        print("Mean estimate", p,np.mean(estimates[p]))  
    return nodes


if __name__ == "__main__":
    G = nx.read_edgelist("./H-I-05.tsv")
    E = list(G.edges())

    Gp = G.subgraph(max(nx.connected_components(G)))
    print("AVEERAGE DEGREE", np.mean([Gp.degree(i) for i in Gp.nodes()]))

    nodes = _mean_estimate(G)


    print("induced subgraph")
    Ginducded =G.subgraph(nodes)
    # nodes = _mean_estimate(Ginducded)
    for p in np.arange(0.1,1.0,0.1):
        estimates ={}
        for _ in range(100):
            nodes = np.random.choice(G.nodes(), size=int(p*len(G)), replace=False)
            H = G.subgraph(nodes)
            estimates[p] = estimates.get(p,[]) + [np.mean([H.degree(v) for v in nodes])]
        print("Mean estimate", p,np.mean(estimates[p]))  

    print("incident subgraph")
    Gincident = G.edge_subgraph(G.edges())
    for p in np.arange(0.1,1.0,0.1):
        estimates ={}
        for _ in range(100):
            edges = np.random.choice(len(E), size=int(p*len(E)), replace=False)
            edges = ([E[e] for e in edges])
            H = G.edge_subgraph(edges)
            estimates[p] = estimates.get(p,[]) + [np.mean([H.degree(v) for v in H.nodes()])]
        print("Mean estimate", p,np.mean(estimates[p]))  
    
    # nodes = _mean_estimate(Gincident)


    print("AVEERAGE DEGREE", np.mean([Gp.degree(i) for i in Gp.nodes()]))



    print("DEGREEE DISTRIBUTION")

    

    print("DIAMETER", nx.diameter(Gp))

    print("GLOBAL CLUSTERING COEFFICIENT", nx.transitivity(Gp))

