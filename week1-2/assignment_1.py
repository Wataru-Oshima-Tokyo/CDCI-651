import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

# erdos_renyi graph generator
def _erdos_renyi_graph(n, p):
    # make a graph
    G = nx.Graph()
    # adding nodes according to n
    nodes =[]
    for a in range(n):
        nodes.append(a)
    G.add_nodes_from(nodes)
    #number of pairs
    # pairs = math.comb(n,2) # n*(n-1)/2
    # print(pairs)
    # adding edges
    edges =[]
    for i in range(n):
        degree = 0
        for j in range(i+1,n): # from the node itself +1 to the last node
            if np.random.rand() <p: # if the random number is bigger than input p then
                edges.append((i,j)) # make an edge between them
    aveDegree = 2*len(edges)/n #<k>
    # print("average of degree is:", aveDegree)
    G.add_edges_from(edges)
    #the size of giantcomponent
    Gcc = len(max(nx.connected_components(G), key=len))
    # print(Gcc)
    return G,Gcc,aveDegree


    
# making a graph

def question4():
    # G = nx.erdos_renyi_graph(8, 1/2)
    G1 = _erdos_renyi_graph(10,0.5)
    G2 = _erdos_renyi_graph(200,0.05)
    G3 = _erdos_renyi_graph(500,0.05)
    position1 = nx.circular_layout(G1)
    position2 = nx.circular_layout(G2)
    position3 = nx.circular_layout(G3)
    nx.draw(G1, position1,with_labels = True)
    plt.show()
    nx.draw(G2, position2,with_labels = True)
    plt.show()
    nx.draw(G3, position3,with_labels = True)
    plt.show()
    # visualizing the graph
    # nx.draw(G, with_labels = True) 
    # plt.show()

def question5():
    n = 1000 # number of nodes
    numOfGraphs =10 # number of graphs
    _p = 0.002 # reasonable probability 
    x =[]
    y =[]
    k =[]
    c =[]
    for a in range(numOfGraphs):
        p = np.random.uniform(0,_p)
        G, Gcc, avgD = _erdos_renyi_graph(n,p)
        x.append(avgD)
        y.append(Gcc/n)
        if avgD >1:
            k.append("Supercritiacal regime")
        elif avgD <1:
            k.append("Subcritical regime")
        else:
            k.append("Crital point")
        if Gcc/n ==1:
            c.append(True)
        else:
            c.append(False)
    # print(x,y)
    plt.scatter(x,y)
    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(x, y, 1)
    line = [m*_x +b for _x in x]
    #use red as color for regression line
    plt.plot(x, line, color='red')
    plt.xlabel('<k>')
    plt.ylabel('(size of giant comp)/number of vertices')
    #Showing the result for each graph
    for i in range(numOfGraphs):
        print("The graph ", end="")
        print(i+1)
        print(k[i])
        if c[i]:
            print("connected regime\n")
        else:
            print("not connected regime\n")
    plt.show()

def question6_1():
    # US airport networks (2010)
    n = 1574 # number of nodes
    E = 28236 # number of edges
    p = E/math.comb(n,2) # probability
    G = nx.gnp_random_graph(n,p, directed=False)
    degrees = [G.degree(i) for i in G.nodes()]
    print(degrees)
    print(G.nodes())
    # plot with various axes scales
    plt.figure()    
    plt.subplot(221)
    plt.scatter(degrees, G.nodes())
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('linear')

    plt.subplot(224)
    plt.scatter(degrees, G.nodes())
    plt.yscale("log")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('log')
    #Showing the result for each graph
    plt.show()

def question6_2():
    # US airport networks (2010)
    n = 1574 # number of nodes
    E = 28236 # number of edges
    p = E/math.comb(n,2) # probability
    # G = nx.gnp_random_graph(n,p, directed=False)
    numOfGraphs = 10 # number of graphs
    x =[]
    y =[]
    k =[]
    c =[]
    Graphs =[]
    #generate 10 ER graphs with n and e
    print(p)
    for a in range(numOfGraphs):
        G, Gcc, avgD = _erdos_renyi_graph(n,p)
        x.append(avgD)
        y.append(Gcc/n)
        Graphs.append(G)
    degrees =[]

    # get all the degrees for each graph
    for i in range(len(Graphs)):
        degrees.append([Graphs[i].degree(j) for j in Graphs[i].nodes()])
    
    avgDegrees =[]
    #take the average of each degree in 10 graphs such as (degree[1][0] + degree[2][0]) /2 ...
    for i in range(len(Graphs[0].nodes())):
        total =0
        for j in range(len(Graphs)):
            total += degrees[j][i]
        avgDegrees.append(total/len(Graphs))

    # expected and average degree distribution
    #Since a degree is equaly treated, there is not weight to consider
    Ep =0
    for i in avgDegrees:
        Ep += i/len(Graphs[0].nodes())
    # Ep += [i/len(Graphs[0].nodes()) for i in avgDegrees]
    print("Expected and approximate degree distribution is ", Ep) 
    # plot with various axes scales
    plt.figure()    
    plt.subplot(221)
    plt.scatter(avgDegrees, Graphs[0].nodes())
    plt.yscale("linear")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('linear')

    plt.subplot(224)
    plt.scatter(avgDegrees, Graphs[0].nodes())
    plt.yscale("log")
    plt.xlabel('Degree')
    plt.ylabel('index of nodes')
    plt.title('log')
    #Showing the result for each graph
    plt.show()

if __name__ == "__main__":
    question4()
    question5()
    question6_1()
    question6_2()