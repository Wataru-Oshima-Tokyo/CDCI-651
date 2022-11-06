import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from  collections import Counter 
import time
from multiprocessing import Pool
import platform
import copy


def readDiabetesdata(network_file):
    G = nx.Graph()
    edge_list =[]
    _max=0
    with open(network_file, 'r') as f:
        #skip the first two lines
        f.readline()
        f.readline()
        for line in f:
            line = line.strip().split() #no arg = split by space
            u = int(line[0])
            v = int(line[1])
            if _max <u:
                _max =u
            edge_list.append((u,v))
    G.add_nodes_from([a for a in range(_max)])
    G.add_edges_from(edge_list)
    return G

def simSIR(G, immunization, p):
    print("started funciton")


    N =len(G)
    states = ['S','I','R']

    immunization_methods = ['random', 'high_deg', 'neighbor']

    R0 =3 #reproductive number
    avg_deg = np.mean([G.degree(v) for v in G.nodes()]) # Avg degree
    spread_rate = R0/avg_deg

    t_max = 30 #Time to simulate spread
    
    iterations = 10 #number of iterations to simulate spread

    total_infected_counter = [0]*t_max
    pos = nx.spring_layout(G)

    #Loop to calculate avgs over 10 graphs
    for i in range(iterations):
        print('Started iteration', i)
        
        #Initializing all nodes to non-immunized
        nx.set_node_attributes(G, False, 'immunized')

        if (immunization == immunization_methods[0]):
            #Immunizing a random node
            randomImmunization(G, p)
        if (immunization == immunization_methods[1]):
            #Immunizing the high degree susceptible node
            highDegImmunization(G, p)
        if (immunization ==immunization_methods[2]):
            #Immunizing a random neighbor of a random node
            neighborImmunization(G, p)  
        
        #Initializing all nodes to susceptible 
        nx.set_node_attributes(G, states[0], 'state')

        # List to track the number of infetecetd at each time step
        infected_counter =[0]*t_max
        infected = set()
        susceptible = {v for v in G.nodes()}
        recoverd = set()   

        #Randomly picking a non-immunized node to be source code
        source = np.random.choice(G.nodes())
        while (G.nodes[source]['immunized'] == True):
            source = np.random.choice(G.nodes())
        #Updating S-I sets, counter, attributes
        infected.add(source)
        susceptible.remove(source)
        nx.set_node_attributes(G,{source: {'state': states[1]}})
        infected_counter[0] +=1
        total_infected_counter[0] += infected_counter[0]

        #Plotting network on last itration

        if i == 9:
            plotNetwork(G, pos, 0)

        for t in range(1, t_max):
            #Copy of infetected set
            curr_infected = infected.copy()

            # Updating infected counter
            infected_counter[t] = infected_counter[t-1]

            # Loop through infeceted nodes
            for v in curr_infected:
                # Looping through neighbors of infected nodes
                for u in nx.neighbors(G,v):
                    #If u is susceptible and the prob. implies infection of node u
                    if np.random.rand() < spread_rate and G.nodes[u]['state'] == states[0] and G.nodes[u]['immunized'] != True :
                        #Updating node u to infected state
                        infected.add(u)
                        susceptible.remove(u)
                        nx.set_node_attributes(G, {u: {'state': states[1]}})
                        infected_counter[t] +=1 #Incrementing infected counter

            # for loop to update infected nodes are recovered 
            for v in curr_infected:
                infected.remove(v)
                recoverd.add(v)
                nx.set_node_attributes(G, {v: {'state': states[2]}})
            
            total_infected_counter[t] += infected_counter[t]

            if i == 9:
                plotNetwork(G, pos, t) #Plotting network 
        print("Finished itration", i)
    time_interval = [x for x in range(t_max)] # Storing time interval as a list

    #Calculating avg infected over 10 graphs
    avg_infected_ct = [x/iterations for x  in total_infected_counter]

    avg_fract_inf = [x/N for x in avg_infected_ct]

    print(avg_fract_inf[t_max-1])

    return [time_interval, avg_fract_inf]

def plotNetwork(G, pos, t):
    # nx.draw(G,pos=pos, with_labels = True)
    # plt.show()
    pass

def randomImmunization(G, p):
    #randomly pick nodes and the number of nodes should be decided by the given probabilities
    observed_nodes = np.random.choice(G.nodes(), size=int(p*len(G)), replace=False)
    for v in observed_nodes:
        G.nodes[v]["immunized"] = True

def highDegImmunization(G, p):
    #sort nodes based off the degree and pick the nodes from the top
    l =sorted(G.degree, key=lambda x: x[1], reverse=True)
    for i in range(int(p*len(G))):
        G.nodes[l[i][0]]["immunized"] = True

def neighborImmunization(G, p):
    #randomly pick the nodes and the number of nodes should be dicided by the given probabilites
    #after that randomly pick one of the neighbors
    observed_nodes = np.random.choice(G.nodes(), size=int(p*len(G)), replace=False)
    for v in observed_nodes:   
        try:                  
            neighbor = np.random.choice(list(G.neighbors(v)), size=1, replace=False)
            G.nodes[neighbor[0]]["immunized"] = True
        except:
            print("error")

def readDiabetesdata(edgeFile, labelFile):
    G = readDiabetesLabels(labelFile)

    with open(edgeFile, 'r') as f:
        #skip the first two lines
        f.readline()
        f.readline()
        for line in f:
            line = line.strip().split() #no arg = split by space
            u = line[1].split(":")[1]
            v = line[-1].split(":")[1]
            if u ==v: #avoid self loop
                continue
            nodes=[]
            for i in G.edges(u):
                nodes.append(i[1])
            #erase self loop and duplicatd edges
            if v not in nodes:
                G.add_edge(u,v)
            
    return G

def readDiabetesLabels(labelFile):
    G = nx.Graph()

    with open(labelFile, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            line = line.strip().split()
            u = line[0]
            label = line[1].split("=")[1]
            # print(label)
            G.add_node(u,label=label)

    return G

def _stublist(degSeq):
    L =[]
    for i in range(len(degSeq)):
        for j in range(degSeq[i]):
            L.append(i)
    return L

def configModel(degSeq):
    G = nx.Graph()
    nodes =[]
    for a in range(len(degSeq)):
        nodes.append(a)
    G.add_nodes_from(nodes)
    edges =[]
    stublist = _stublist(degSeq)
    np.random.shuffle(stublist)
    n = len(stublist)
    half = n//2
    _out, _in = stublist[:half], stublist[half:]
    G.add_edges_from(zip(_out, _in))
    return G

def get_dict_inOrder(dict):
    keys = list(dict.keys())
    temp = []
    for i in keys:
        temp.append(int(i))
    temp.sort()
    keys =[]
    for i in temp:
        keys.append(str(i))
    return ([dict[i] for i in keys])

def readFile(file_name):
    G = nx.Graph()
    labels = []
    with open(file_name, 'r') as f:
        # id name, neighbors
        w =0
        for line in f:
            line = line.strip().split() #no arg = split by space
            label = line[1].split(",")[0]
            print(label)
            labels.append(label)
            G.add_node(line[0], label=label)
            for i in range(len(line)-2):
                G.add_edge(line[0],line[i+2])
            w+=1

    return G, labels

def question_3_1():
    file_path = "/Users/wataruoshima/CSCI651/week9-10/medici_network.adjlist.txt"
    G, labels = readFile(file_name=file_path)
    position1 = nx.circular_layout(G)
    # nx.draw(G,pos=position1, with_labels = True)
    # nx.draw_networkx_labels(G, pos=position1,labels=labels)
    # plt.show()
    #show the degree centrality for each vertex
    degree_centrality = nx.degree_centrality(G)
    # harmonic centrality
    harmonic_centrality = nx.harmonic_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    for i in G.nodes():
        print("Node: %d (%s) has %lf degree centrality, %lf harmonic centrality, %lf eigenvector centrality, %lf betweenness_centrality" %(int(i), labels[int(i)], degree_centrality[i], harmonic_centrality[i],eigenvector_centrality[i], betweenness_centrality[i]))
    #For each measure, make a table with the families ranked by importance. Show these tables side by side
    real_harmonic_centrality = get_dict_inOrder(harmonic_centrality)

    #get the degree seaquence
    _degreeSeq ={}
    for i in G.nodes():
        _degreeSeq[i] = G.degree[i] 
    degreeSeq = get_dict_inOrder(_degreeSeq)
    random_harmonic_centrality = {}
    total_harmonic_centrality =0
    counter=0
    for i in range(10000):
        G = configModel(degSeq=degreeSeq)
        harmonic_centrality = nx.harmonic_centrality(G)
        for i in G.nodes():
            if i in random_harmonic_centrality:
                random_harmonic_centrality[i].append(harmonic_centrality[i])
            else:
                random_harmonic_centrality[i] = [harmonic_centrality[i]]
            total_harmonic_centrality += harmonic_centrality[i]
            counter+=1
    
    mean_harmonic_centrality = total_harmonic_centrality/counter
    stds =[]
    for i in random_harmonic_centrality:
        stds.append(np.std(random_harmonic_centrality[i]))
        random_harmonic_centrality[i] = real_harmonic_centrality[i] - np.mean(random_harmonic_centrality[i])
    print(random_harmonic_centrality)
    values = list(random_harmonic_centrality.values())
    print(mean_harmonic_centrality)

    print(labels)
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(labels, values, color="red",label='Values from Degree seq.',alpha=0.3, edgecolors='none')
    plt.errorbar(labels, values, yerr=stds, fmt="o", color="red")
    # plt.scatter(labels, real_harmonic_centrality,color="green",label='Actual data',alpha=0.3, edgecolors='none')
    # plt.yscale("log")
    plt.xlabel('Families')
    plt.ylabel('Harmonic centrality - mean of Harmonic centrality')
    plt.title('Harmonic centrality distribution')
    plt.legend()
    plt.show()

def question_3_2():
    edgeFile = "./pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"
    labelFile = "./pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab"
    G = readDiabetesdata(edgeFile=edgeFile, labelFile=labelFile)
    # G =_erdos_renyi_graph(1000, 0.001)
    print(len(G.nodes()))
    # position1 = nx.circular_layout(G)
    # nx.draw(G,pos=position1, with_labels = True)
    # plt.show()
    estimatiion =[]
    std_error = []
    ps=[]
    for p in np.arange(0.1,1.0,0.1):
        temp_estimate =[]
        for _ in range(10):
            Gp = copy.deepcopy(G)
            label_nodes = {}
            unobserved_nodes =[]
            observed_nodes = np.random.choice(Gp.nodes(), size=int(p*len(Gp)), replace=False)
            
            #initialize the label of all the nodes except the randomly chosen nodes
            for v in Gp.nodes():
                if v in observed_nodes:
                    # print("here")
                    continue
                else:
                    label_nodes[v] = int(Gp.nodes[v]["label"]) #store the previous label so that we can compare if the predicted label is the same the actual one
                    Gp.nodes[v]["label"] = "0"
                    unobserved_nodes.append(v)
            success_counter=0
            total_counter=0
            print("observed list: ", len(observed_nodes))
            print("unobserved list: ", len(unobserved_nodes))
            start = time.time()
            for v in unobserved_nodes:
                total_counter+=1
                labels = [0,0,0]
                # print(v, "neighbors:")
                for u in Gp.neighbors(v): #check all the neighbors of v to use guild by association heuristic
                    _label = nx.get_node_attributes(Gp, "label")[u]
                    if int(_label) == 1:
                        labels[0] +=1
                    elif int(_label) == 2:
                        labels[1] +=1
                    elif int(_label) == 3:
                        labels[2] +=1
                _max =[]
                index = labels.index(max(labels))
                _max.append(index+1)
                for i in range(len(labels)):
                    if index == i:
                        continue
                    elif labels[i] == labels[index]:
                        _max.append(i+1)
                prob = 1.0/len(_max)
                prob_list = [prob] * len(_max)
                node_label = np.random.choice(_max, p =prob_list)
                # print(node_label)
                if node_label == label_nodes[v]:
                    # print("same!")
                    success_counter+=1
                else:
                    # print("different...")
                    pass
                # time.sleep(3)
            print(p, "prob:", float(success_counter/total_counter))
            temp_estimate.append(float(success_counter/total_counter))
            end = time.time()
            print(i, " total time:",(end-start))
        estimatiion.append(np.mean(temp_estimate))
        std_error.append(np.std(temp_estimate))
        ps.append(p)
    plt.figure()    
    # plt.subplot(221)
    plt.scatter(ps, estimatiion, color="red",label='Induced subgraph',alpha=0.3, edgecolors='none')
    plt.errorbar(ps, estimatiion, yerr=std_error, fmt="o", color="red")
    # plt.yscale("log")
    plt.xlabel('Probabilities')
    plt.ylabel('Average fraction of correct guesses')
    plt.title('Guilt by association heuristic')
    plt.legend()
    plt.show()

def question_3_3():
    network_file = "./facebook-wosn-wall/out.facebook-wosn-wall"
    G = readDiabetesdata(network_file)
    # G = nx.barabasi_albert_graph(1000,3)
    print(G.number_of_nodes(), G.number_of_edges())
    probablies = [0.1,0.3, 0.5,0.7,0.9]
    colors = ["red", "green", "blue", "yellow","indigo"]
    immunization_methods = ['random', 'high_deg', 'neighbor']
    subplot_index = [221,222,223]

    plt.figure()    
    for i,immunization_method in enumerate(immunization_methods):    
        pool = Pool() # make a multiprocessing
        plt.subplot(subplot_index[i])
        data_list =[]
        for j,p in enumerate(probablies):
            data = [G, immunization_methods[i], p]
            data_list.append(data)
        returned_data = pool.starmap(simSIR, data_list)
        for j,p in enumerate(probablies):
            plt.scatter(returned_data[j][0], returned_data[j][1], color=colors[j],label=p,alpha=0.3, edgecolors='none')
        pool.close()
        print(immunization_method, " finished")
        plt.xlabel('Time')
        plt.ylabel('It/n (the fraction of infected nodes)')
        plt.title(immunization_methods[i])
    plt.legend()
    plt.show() 

if __name__ == "__main__":
    question_3_1()
    question_3_2()
    question_3_3()