
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

if __name__ == "__main__":
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