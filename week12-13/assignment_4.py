import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import combinations_with_replacement, product
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from collections import defaultdict
from networkx.utils.mapped_queue import MappedQueue
from multiprocessing import Pool
def _modularity(G, C):
    """
    (5 pts) Write a function modularity(G, C) to calculate the modularity of the graph G where
    C is a dictionary with nodes as keys and community number as values.
    """
    g = 1
    numOfGroups = len(nx.unique(C['values'])) # check how many communities in the graph
    comm = [None] * numOfGroups
    for v in G.nodes():
        for i in range(numOfGroups):
            if v['values'] == i:
                comm[i].append(v) #splitting the nodes to each group
    
    m = G.edges()
    visited =[]
    links = [0] * numOfGroups
    sumOfDegrees = [0] * numOfGroups
    for v in G.nodes():
        for u in G.neighbors(v): # check all the neighbors 
            for n in range(len(comm)): # if it is in the community
                if u in comm[n]:
                    sumOfDegrees[n] += 1
                    if u not in visited: # if the node is not visited, whcih means avoiding the same egge multiple times
                        links[n] += 1    
        visited.append(v)
    Q =[]
    for i in range(numOfGroups):
        Q += (links[i]/m -g*pow(sumOfDegrees[i]/(2*m),2))
    return Q


def _greedyModularity(G):

    N = G.number_of_nodes()
    weight =None
    resolution = 1
    m = len(G.edges())
    q0 = 1 / m
    a = b = {node: degree * q0 * 0.5 for node, degree in G.degree()}
    queue_dict = defaultdict(lambda: defaultdict(float))
    for u, v in G.edges():
        if u == v:
            continue
        queue_dict[u][v] += 1
        queue_dict[v][u] += 1

    for u, nbrdict in queue_dict.items():
        for v, wt in nbrdict.items():
            queue_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])

    queue_heap = {u: MappedQueue({(u, v): -dq for v, dq in queue_dict[u].items()}) for u in G}
    H = MappedQueue([queue_heap[n].heap[0] for n in G if len(queue_heap[n]) > 0])

    communities = {n: frozenset([n]) for n in G}
    yield communities.values()

    while len(H) > 1:
        try:
            negdq, u, v = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        queue_heap[u].pop()
        if len(queue_heap[u]) > 0:
            H.push(queue_heap[u].heap[0])
        if queue_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            queue_heap[v].remove((v, u))
            if len(queue_heap[v]) > 0:
                H.push(queue_heap[v].heap[0])
        else:
            queue_heap[v].remove((v, u))
        communities[v] = set(communities[u] | communities[v])
        del communities[u]

        u_nbrs = set(queue_dict[u])
        v_nbrs = set(queue_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        for w in all_nbrs:
            if w in both_nbrs:
                queue_vw = queue_dict[v][w] + queue_dict[u][w]
            elif w in v_nbrs:
                queue_vw = queue_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:  
                queue_vw = queue_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            for row, col in [(v, w), (w, v)]:
                queue_heap_row = queue_heap[row]
                queue_dict[row][col] = queue_vw
                if len(queue_heap_row) > 0:
                    d_oldmax = queue_heap_row.heap[0]
                else:
                    d_oldmax = None
                d = (row, col)
                d_negdq = -queue_vw
                if w in v_nbrs:
                    queue_heap_row.update(d, d, priority=d_negdq)
                else:
                    queue_heap_row.push(d, priority=d_negdq)
                if d_oldmax is None:
                    H.push(d, priority=d_negdq)
                else:
                    row_max = queue_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)

        for w in queue_dict[u]:
            dq_old = queue_dict[w][u]
            del queue_dict[w][u]
            if w != v:
                for row, col in [(w, u), (u, w)]:
                    queue_heap_row = queue_heap[row]
                    d_old = (row, col)
                    if queue_heap_row.heap[0] == d_old:
                        queue_heap_row.remove(d_old)
                        H.remove(d_old)
                        if len(queue_heap_row) > 0:
                            H.push(queue_heap_row.heap[0])
                    else:
                        queue_heap_row.remove(d_old)

        del queue_dict[u]
        queue_heap[u] = MappedQueue()
        a[v] += a[u]
        a[u] = 0

        yield communities.values()


def question1():
    file_path = "/Users/wataruoshima/CSCI651/week12-13/lesmis/lesmis.gml"
    G = nx.read_gml(file_path)
    p= [1,1]
    q = [0.5,2]
    k = [6,3]
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**    
    for i in range(2):
        node2vec = Node2Vec(G,dimensions=20, walk_length=15, num_walks=500, p=p[i],q=q[i], workers=1, seed=1234)
        model = node2vec.fit(window=15, min_count=1)
        embedding = [model.wv[str(v)] for v in sorted(G.nodes())]
        kmeans_model = KMeans(n_clusters=k[i]).fit(embedding)
        yHat =kmeans_model.labels_
        print("With k=%d, p=%d, and q=%.1f, the number of groups is %d" %(k[i],p[i],q[i],len(np.unique(yHat))))

        pos = nx.spring_layout(G)
        # nx.draw_networkx(G, pos=pos, node_size=100, font_size=5, width=0.5)
        file_name = 'node2vec_figure1_' + str(i) + '.png'
        # plt.savefig(file_name, bbox_inches="tight")
        # plt.clf()
        print('silhouette coefficient:', metrics.silhouette_score(embedding,yHat))
        tsne = TSNE(n_components=2, init='pca')
        pos = tsne.fit_transform(np.array(embedding))
        pos = {v:pos[i] for i,v in enumerate(sorted(G.nodes()))}
        nx.draw_networkx(G, pos=pos, edgelist=[], alpha=0.5, node_size=100, font_size=5,
                        nodelist=sorted(G.nodes()), node_color=yHat)
        plt.savefig(file_name, bbox_inches='tight')
        plt.clf()

def question3():
    G = nx.karate_club_graph()
    file_name = 'karate_club.png'
    _c = _greedyModularity(G)
    communities = next(_c)

    while len(communities) > 1:
        try:
            dq = next(_c)
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            while len(communities) > G.number_of_nodes():
                comm1, comm2, *rest = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            break

        if dq < 0 and len(communities) <= G.number_of_nodes():
            break
        communities = next(_c)

    communities = sorted(communities, key=len, reverse=True)
    for i in range(len(communities)):
        print("The group %d is " %i,communities[i])
    
def question5():
    attempts = 100
    prob_total =0
    for i in range(attempts):
        total =0
        G = nx.erdos_renyi_graph(1000,0.1)
        for i in G.nodes():
            if G.degree(i)  == 95:
                total +=1
        prob = (total/1000) *100
        print("The probability of the nodes having 95 degrees is %lf percent" %(prob))
        prob_total += prob
    print("The mean probability of the nodes having 95 degrees in 100 attempts is %lf pecent" %(prob_total/attempts))


def question6_helper(p,index):
    print("Index %d started!" %(index))
    mean_gcc =0
    for i in range(10):
        G = nx.erdos_renyi_graph(1000, p)
        Gcc = len(max(nx.connected_components(G), key=len))
        # print(Gcc)
        mean_gcc +=Gcc
        # print(mean_gcc)
    print("Index %d Finished!" %(index))
    return mean_gcc/10

def question6():
    """
    Since the number of node is 1000, 
    Subcritical regime is p<0.001
    Critical point is p = 0.001
    Superctical point is p>0.001
    Connected regime is p>ln(1000)/1000 {0.00690}
    
    If the probabilty is less than 1/N which is 0.0001, they are mostly separetd but it exceeds the critical point 
    the components are rapidly connecetd and become one giant component.
    """
    
    ps =[p for p in np.arange(0.0,0.01,0.0001)]
    data_list =[]
    for j,p in enumerate(ps):
        data = [p, j]
        data_list.append(data)
    pool = Pool()
    returned_data = pool.starmap(question6_helper, data_list)
    pool.close()
    values =[]
    for i in range(len(returned_data)):
        values.append(returned_data[i]/1000)
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(ps, values, color="red",label='Four regimes',alpha=0.3, edgecolors='none')
    # plt.errorbar(ps, harmonic_centrality_total, yerr=std_error, fmt="o", color="red")
    # plt.yscale("log")
    plt.xlabel('Probabilities')
    plt.ylabel('Ng/N')
    plt.title('Question ')
    plt.legend()
    plt.show()

    


def question7_helper(G, mean_harmonic_centrality, index):
    print("Index %d started!" %(index))
    meanofHarmonic_centrality =0
    
    harmonic_centrality = nx.harmonic_centrality(G)
    for i in G.nodes():
        meanofHarmonic_centrality += harmonic_centrality[i]
    mean_harmonic_centrality = meanofHarmonic_centrality/1000
    print("Index %d Finished!" %(index))
    return mean_harmonic_centrality
    

def question7():
    ps =[p for p in np.arange(0.05,0.95,0.05)]
    harmonic_centrality_array = [0] * len(ps)
    data_list =[]
    for j,p in enumerate(ps):
        G = nx.erdos_renyi_graph(1000, p)
        data = [G, harmonic_centrality_array[j], j]
        data_list.append(data)
    pool = Pool()
    returned_data = pool.starmap(question7_helper, data_list)
    pool.close()
    values =[]
    for i in range(len(returned_data)):
        values.append(returned_data[i]/np.mean(returned_data))
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(ps, values, color="red",label='harmonic centrality',alpha=0.3, edgecolors='none')
    # plt.errorbar(ps, harmonic_centrality_total, yerr=std_error, fmt="o", color="red")
    # plt.yscale("log")
    plt.xlabel('Probabilities')
    plt.ylabel('Average fraction of harmonic centrality')
    plt.title('Question 7')
    plt.legend()
    plt.show()


def question8_helper(n,p,index):
    print("Index %d started!" %(index))
    mean_degree_distribution=0
    # print(params)
    for i in range(100):
        G = nx.extended_barabasi_albert_graph(n,1,p,0)       
        degrees=[G.degree(i) for i in G.nodes()]
        degree_distribution = np.mean(degrees)
        # print(degree_distribution)
        mean_degree_distribution +=degree_distribution
        # print(mean_gcc)
    print("Index %d Finished!" %(index))
    return mean_degree_distribution/100

def question8():
    """
    Mean degree distribution is 2.018014: m=1
    """
    n=1000
    data_list =[]
    ps =[p for p in np.arange(0.001,0.02,0.001)]
    for j, p in enumerate(ps):
        data = [n, p ,j]
        data_list.append(data)
    pool = Pool()
    returned_data = pool.starmap(question8_helper, data_list)
    pool.close()
    mean_degree_distribution=np.mean(returned_data)
    print("Mean degree distribution is %lf" %(mean_degree_distribution))


def question9_helper(n,p,index):
    print("Index %d started!" %(index))
    mean_degree_distribution=0

    attempts=10
    mean_gcc=0
    for i in range(attempts):
        G = nx.extended_barabasi_albert_graph(n,1,p,0)  
        Gcc = len(max(nx.connected_components(G), key=len))
        print(Gcc)
        mean_gcc +=Gcc
        # print(mean_gcc)
    print("Index %d Finished!" %(index))
    return mean_gcc/attempts


def question9():
    n=1000
    data_list =[]
    ps =[p for p in np.arange(0.001,0.01,0.0001)]
    for j,p in enumerate(ps):
        data = [n, p, j]
        data_list.append(data)
    pool = Pool()
    returned_data = pool.starmap(question9_helper, data_list)
    pool.close()
    values =[]
    for i in range(len(returned_data)):
        values.append(returned_data[i]/n)
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(ps, values, color="red",label='Four regimes',alpha=0.3, edgecolors='none')
    # plt.errorbar(ps, harmonic_centrality_total, yerr=std_error, fmt="o", color="red")
    # plt.yscale("log")
    plt.xlabel('Probabilities')
    plt.ylabel('Ng/N')
    plt.title('Question 9')
    plt.legend()
    plt.show()

def question10():
    """
    The network is specified by G(n,m). 
    The network starts with m connected states. 
    Nodes with m links are added to the network one by one. 
    m new links each connect to an existing node with a probability proportional to the degree of the existing node. 
    Therefore, the node with the highest degree is more likely to have links. This is called preferential selection. 
    Nodes are added repeatedly until the number of nodes reaches n. 
    The resulting network is a scale-free network.
    """
    
if __name__ == "__main__":
    # question9()
    question9()