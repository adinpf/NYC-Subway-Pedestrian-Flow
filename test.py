import pickle 
import networkx as nx


with open('/Users/chrisparvankin/Desktop/CS1470/NYC-Subway-Pedestrian-Flow/data/transport_ridership.pkl', 'rb') as file:  
    d = pickle.load(file)

with open('/Users/chrisparvankin/Desktop/CS1470/NYC-Subway-Pedestrian-Flow/data/subway_network.pkl', 'rb') as file:  
    e = pickle.load(file)


print(d.iloc[0])
first_node = list(e.nodes())[0]    
first_node_data = e.nodes[first_node]  

print("First node:", first_node)
print("First node attributes:", first_node_data)