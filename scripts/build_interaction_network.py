import argparse
import json
import re
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
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
    df = pd.read_csv(csv_filepath, encoding='utf-8')
    #df_reduced_characters = df[any(w in df['pony'].str for w in CHARACTER_STOPWORDS)]
    most_frequent_characters = df['pony'].value_counts()[:101].index.tolist()
    most_frequent_characters = [c.lower() for c in most_frequent_characters]
    # df = df[df['pony'] in most_frequent_characters]

    character_interactions = nx.Graph()
    character_interactions.name = "Character Interactions"
    
    # df.reset_index()
    previous_character = ''
    for index, row in df.iterrows():
        character = row['pony'].lower()
        
        if is_valid_interaction(character, previous_character, most_frequent_characters):
            if character_interactions.has_edge(character, previous_character):
                character_interactions[character][previous_character]['weight'] += 1
            else:
                character_interactions.add_edge(character, previous_character, weight=1)
        
        previous_character = character
        
    return character_interactions


def draw_graph(g: nx.Graph, filename: str):
    print(f"draw_graph: Drawing graph '{g.name}'...")
    f = plt.figure(figsize=(70, 70), dpi=80)

    pos = nx.spring_layout(g, k=(16 / np.sqrt(g.order())))
    node_degrees = dict(g.degree)
    node_sizes = [deg * 60 for deg in node_degrees.values()]
    nx.draw_networkx(g, pos, font_size=8, nodelist=node_degrees.keys(), node_size=node_sizes)

    print("draw_graph: Scaling an coloring edges...")
    weights = nx.get_edge_attributes(g, 'weight')
    max_weight = max(weights.values())
    min_weight = min(weights.values())

    c_norm  = colors.Normalize(vmin=min_weight, vmax=max_weight)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=plt.get_cmap('jet'))

    for edge in g.edges(data='weight'):
        weight = edge[2]
        width = (weight / max_weight) * 25
        color = scalar_map.to_rgba(weight)
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=width, edge_color=color)
        
    f.savefig(filename)
    print(f"draw_graph: Graph '{g.name}' has been saved to '{filename}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The path to the input script csv file.")
    parser.add_argument("-o", "--output", required=True, help="The path to the output network json file.")
    args = parser.parse_args()
    
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    g = get_interactions_graph(args.input)
    draw_graph(g, "mlp_character_interactions.png")
    
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     json.dump(g, file, indent=4)


if __name__ == '__main__':
    main()
