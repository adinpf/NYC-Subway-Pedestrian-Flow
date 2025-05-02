from geopy.distance import geodesic
import pandas as pd
import networkx as nx
import numpy as np

def calculate_distance(coord1, coord2):
    '''
    calculates the geodesic distance in kilometers between two coordinate pairs (latitude, longitude)
    for missing coordinates return infinity distance
    '''
    if pd.isna(coord1[0]) or pd.isna(coord1[1]) or pd.isna(coord2[0]) or pd.isna(coord2[1]):
        return float('inf') # Treat missing coords as infinitely far
    try:
        return geodesic(coord1, coord2).km
    except ValueError:
        return float('inf') # Handle potential errors in geopy

def find_station_id(graph: nx.Graph, stop_id: str, stop_id_to_sc_id: dict, stop_df: pd.DataFrame):
    '''
    find node ID in the graph that is geographically closest to the given GTFS stop_id
    use coordinates from stop_df and graph node attributes
    updates the stop_id_to_sc_id dictionary accordingly
    
    return the matching graph node ID or None if no match is found
    '''
    stop_row = stop_df[stop_df["stop_id"] == str(stop_id)]
    if stop_row.empty:
        print(f"Warning: Stop ID {stop_id} not found in stop_df.")
        return None
    stop_coords = (stop_row["stop_lat"].iloc[0], stop_row["stop_lon"].iloc[0])
    min_distance = np.inf
    node_match = None

    for node in graph.nodes:
        node_coords = graph.nodes[node].get('coordinates') # Use .get for safety
        if node_coords is None:
            continue # Skip nodes without coordinates

        curr_distance = calculate_distance(stop_coords, node_coords)

        if curr_distance < min_distance:
            min_distance = curr_distance
            node_match = node

    # Only update if a match was found
    if node_match is not None:
        stop_id_to_sc_id[stop_id] = node_match

    return node_match

def add_edge(graph: nx.Graph, stop_ids: tuple[str, str, int], stop_id_to_sc_id: dict, stop_df: pd.DataFrame):
    '''
    add an edge to the graph between the nodes corresponding to the two stop IDs in stop_ids
        - edge includes 'distance' and 'num_connections' attributes
    '''
    stop1_id = stop_ids[0]
    stop2_id = stop_ids[1]
    num_connections = stop_ids[2]

    try:
        station_id1 = stop_id_to_sc_id[stop1_id]
    except KeyError:
        # Find station_id1 and add it to the dictionary
        station_id1 = find_station_id(graph, stop1_id, stop_id_to_sc_id, stop_df)

    try:
        station_id2 = stop_id_to_sc_id[stop2_id]
    except KeyError:
        # Find station_id2 and add it to the dictionary
        station_id2 = find_station_id(graph, stop2_id, stop_id_to_sc_id, stop_df)

    # Check if nodes exist before adding edge
    if station_id1 is None or station_id2 is None:
        print(f"Warning: Could not find matching nodes for stop IDs {stop1_id} or {stop2_id}. Skipping edge.")
        return

    # Ensure nodes actually exist in the graph (find_station_id might return None)
    if not graph.has_node(station_id1) or not graph.has_node(station_id2):
         print(f"Warning: Node {station_id1} or {station_id2} not found in graph. Skipping edge.")
         return

    if not(graph.has_edge(station_id1, station_id2)):
        # Calculate distance between stations by getting latitude and longitude
        coord1 = graph.nodes[station_id1].get('coordinates')
        coord2 = graph.nodes[station_id2].get('coordinates')
        if coord1 is None or coord2 is None:
            print(f"Warning: Missing coordinates for nodes {station_id1} or {station_id2}. Skipping distance calculation.")
            distance = float('inf')
        else:
            distance = calculate_distance(coord1, coord2)
        graph.add_edge(station_id1, station_id2, distance=distance, num_connections=num_connections)

def add_edges(graph, stop_pairs, stop_df):
    '''
    iterate through a list of stop pairs and add_edge for each pair to add edges to the graph
    '''
    stop_id_to_sc_id = {} # dict to track pairs
    for pair in stop_pairs:
        add_edge(graph, pair, stop_id_to_sc_id, stop_df)



