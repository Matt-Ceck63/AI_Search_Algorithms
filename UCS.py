# -*- coding: utf-8 -*-
"""Coursework1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HV2a8paT4mSu-QXDAMcjZajMmKH2w5rk
"""

### QUESTION 1 ###

import networkx as nx
import matplotlib.pyplot as plt
import json

# CODE FROM LABS
def load_graph_from_file(filename):
  with open(filename) as graph_file:
    dict_graph = json.load(graph_file)
    return nx.Graph(dict_graph)

def show_weighted_graph(networkx_graph, node_size, font_size, fig_size):
  # Allocate the given fig_size in order to have space for each node
  plt.figure(num=None, figsize=fig_size, dpi=80)
  plt.axis('off')
  # Compute the position of each vertex in order to display it nicely
  nodes_position = nx.spring_layout(networkx_graph) 
  # You can change the different layouts depending on your graph
  # Extract the weights corresponding to each edge in the graph
  edges_weights  = nx.get_edge_attributes(networkx_graph,'weight')
  # Draw the nodes (you can change the color)
  nx.draw_networkx_nodes(networkx_graph, nodes_position, node_size=node_size,  
                         node_color = ["orange"]*networkx_graph.number_of_nodes())
  # Draw only the edges
  nx.draw_networkx_edges(networkx_graph, nodes_position, 
                         edgelist=list(networkx_graph.edges), width=2)
  # Add the weights
  nx.draw_networkx_edge_labels(networkx_graph, nodes_position, 
                               edge_labels = edges_weights)
  # Add the labels of the nodes
  nx.draw_networkx_labels(networkx_graph, nodes_position, font_size=font_size, 
                          font_family='sans-serif')
  plt.axis('off')
  plt.show()

# Network map
cities = load_graph_from_file("UK_cities.json")

### My Functions ###

def pop_min(frontier):
  # Returns the node with the minimum cost value from the frontier and removes it from the frontier'''

  temp_node = ()
  min_val = 1000000

  for node in frontier:
    if node[1] <= min_val:
      min_val = node[1] 
      temp_node = node
    
  frontier.remove(temp_node) # frontier is passed by reference

  return temp_node

def get_actions(graph, node):

  # returns all the nodes to the next states from the current state
  # also includes the utility function: cost = distance

  state = node[0]
  cost = node[1]
  path = node[2]
  neighbors = list(graph.neighbors(state))
  edges = nx.edges(graph, [state])
  paths = []
  weights = []

  for n, edge in enumerate(edges):
    weight = graph.get_edge_data(*edge)['weight']
    weights.append(cost+weight) # add cost from previous node
    paths.append(path + [neighbors[n]])  # do [path from previous node + name of neigbour]

  actions = list(zip(neighbors, weights, paths)) # Create new nodes

  return actions

def get_previous_cost(frontier, child_node):

  # gets the cost of the node with the same name already on the frontier
  
  for i, node in enumerate(frontier):
    if node[0] == child_node[0]:
      return [i, node]

def ucs(graph, start_node, goal_node, action_function):

  # algorith applied from AI: a modern approach book
  # print statements used to answer question 1-b
  # returns the goal node which contains the total cost of the path and the path itself

  frontier = [(start_node, 0, [start_node])]
  explored = set()
  # print(frontier)

  while True:
    # print("ITERATION ", i)
    if not frontier:
      return False
    node = pop_min(frontier) # pop node with min cost from frontier
    # print("Popped = ", node) # " From frontier: ", frontier)
    if node[0] == goal_node:
      return node # if state is goal state then return path
    explored.add(node[0])
    # print("Explored = ", explored)
    actions = action_function(graph, node) # ACTION(s, n) return neighbour nodes with relative cost
    for child_node in actions:
      states = [i[0] for i in frontier]
      if child_node[0] not in explored and child_node[0] not in states:
        frontier.append(child_node)
        # print("Frontier modified = ", child_node, " Added")
      elif child_node[0] in states:
        previously_known_cost_node_index, previously_known_cost_node = get_previous_cost(frontier, child_node)
        if child_node[1] < previously_known_cost_node[1]:
          frontier[previously_known_cost_node_index] = child_node
          # print("Frontier modified = ", child_node, " Replaced")

# Tested algorithm
pathb = ucs(cities, "london", "aberdeen", get_actions)

def get_actions_with_time_and_pollution(graph, node):

  # returns all the nodes to the next states from the current state
  # set a constant speed for all paths
  speed = 10 #km/h
  # time = distance/speed expressed in hours
  # cost = time + (0.00001 * (speed**2) * time)

  state = node[0]
  cost = node[1]
  path = node[2]
  neighbors = list(graph.neighbors(state))
  # edges still represent distances
  edges = nx.edges(graph, [state])
  costs = []
  paths = []
  for n, edge in enumerate(edges):
    distance = graph.get_edge_data(*edge)['weight']
    time = distance/speed
    new_cost = time + (0.00001 * (speed**2) * time)
    costs.append(cost + new_cost) # add cost from previous node
    paths.append(path + [neighbors[n]])  # do [path of previous node, name of neigbour]

  actions = list(zip(neighbors, costs, paths))

  return actions

# tested algorithm with new fitness function
pathc = ucs(cities, "london", "aberdeen", get_actions_with_time_and_pollution)

def get_actions_with_supercar(graph, node):

  # returns all the nodes to the next states from the current state

  def calculate_cost(distance):
  
  # utility function, just used to make the code more readable, returns the cost based on distance
    from math import e
    import random
    fine_probability = lambda v, v_lim : 1 - e**(-(v-v_lim))
    v_lim = distance
    if distance > 100:
      speed = 300
    elif distance > 50 and distance <= 100:
      speed = 150
    else:
      speed = distance
    time = distance/speed
    rent_cost = time * 100
    fine_cost = 1000 if random.random() <= fine_probability(speed, v_lim) else 0 # simulate proability using random numbers
    new_cost = rent_cost + fine_cost
    return new_cost

  state = node[0]
  cost = node[1]
  path = node[2]
  neighbors = list(graph.neighbors(state))
  # edges still represent distances
  edges = nx.edges(graph, [state])
  costs = []
  paths = []

  for n, edge in enumerate(edges):
    distance = graph.get_edge_data(*edge)['weight']
    costs.append(cost + calculate_cost(distance)) # add cost from previous node
    paths.append(path + [neighbors[n]])  # do [path of previous node, name of neigbour]

  actions = list(zip(neighbors, costs, paths))

  return actions

# tested algorithm
pathd = ucs(cities, "london", "aberdeen", get_actions_with_supercar)

print(pathb)
print(pathc)
print(pathd)