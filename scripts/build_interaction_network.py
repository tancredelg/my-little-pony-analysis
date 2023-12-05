import argparse
import json
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np


CHARACTER_STOPWORDS = ['others', 'ponies', 'and', 'all']


def is_valid_interaction(character1, character2, most_frequent_characters):
    if character1 == character2:
        return False

    valid_c1 = (not any(w in character1 for w in CHARACTER_STOPWORDS)) and character1 in most_frequent_characters 
    valid_c2 = (not any(w in character2 for w in CHARACTER_STOPWORDS)) and character2 in most_frequent_characters 
    return valid_c1 and valid_c2


def get_interactions_graph(csv_filepath) -> nx.Graph:
    print("Building graph...")
    df = pd.read_csv(csv_filepath, encoding='utf-8')
    most_frequent_characters = df['pony'].value_counts()[:101].index.tolist()
    most_frequent_characters = [c.lower() for c in most_frequent_characters]

    interactions_graph = nx.Graph()
    interactions_graph.name = "Character Interactions"

    previous_character = previous_episode = ''
    for index, row in df.iterrows():
        if row['title'] != previous_episode:
            previous_character = ''
            previous_episode = row['title']

        character = row['pony'].lower()

        if is_valid_interaction(character, previous_character, most_frequent_characters):
            if interactions_graph.has_edge(character, previous_character):
                interactions_graph[character][previous_character]['weight'] += 1
            else:
                interactions_graph.add_edge(character, previous_character, weight=1)

        previous_character = character

    print(f"Graph '{interactions_graph.name}' has been built.")
    return interactions_graph


def draw_graph(g: nx.Graph, filepath: str):
    print("Drawing graph...")
    f = plt.figure(figsize=(280, 280), dpi=30)

    pos = nx.spring_layout(g, k=(24 / np.sqrt(g.order())))
    node_degrees = dict(g.degree(weight='weight'))
    node_sizes = [wdeg * 18 for wdeg in node_degrees.values()]
    nx.draw_networkx(g, pos, font_size=36, nodelist=node_degrees.keys(), node_size=node_sizes)

    weights = nx.get_edge_attributes(g, 'weight')
    max_weight = max(weights.values())
    min_weight = min(weights.values())

    c_norm = colors.Normalize(vmin=min_weight, vmax=max_weight)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=plt.get_cmap('jet'))

    for edge in g.edges(data='weight'):
        weight = edge[2]
        width = (weight / max_weight) * 256
        color = scalar_map.to_rgba(weight)
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=width, edge_color=color)

    f.savefig(filepath)
    print(f"Graph '{g.name}' has been saved to '{filepath}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The path to the input script csv file.")
    parser.add_argument("-o", "--output", required=True, help="The path to the output network json file.")
    parser.add_argument("-n", "--network-output", help="The path to the output network png file.")
    args = parser.parse_args()

    g = get_interactions_graph(args.input)

    if args.network_output:
        draw_graph(g, args.network_output)

    # Extract interaction network dictionary from the graph
    interaction_network = dict(g.nodes)
    for u, v, w in g.edges(data='weight'):
        interaction_network[u][v] = w

    # Save the interaction network dictionary
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(interaction_network, file, indent=4)


if __name__ == '__main__':
    main()
