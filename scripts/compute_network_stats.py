import argparse
import json
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from build_interaction_network import draw_graph


def compute_top_centralities(g: nx.Graph, n: int, with_values: bool = False) -> dict:
    g_distances = {(e1, e2): 1 / weight for e1, e2, weight in g.edges(data='weight')}
    nx.set_edge_attributes(g, g_distances, 'distance')

    degrees = dict(nx.degree_centrality(g))
    weighted_degrees = dict(g.degree(weight='weight'))
    closenesses = dict(nx.closeness_centrality(g, distance='distance'))
    betweennesses = dict(nx.betweenness_centrality(g))

    if with_values:
        return {
            "degree": dict(sorted(degrees.items(), key=lambda i: i[1], reverse=True)[:n]),
            "weighted_degree": dict(sorted(weighted_degrees.items(), key=lambda i: i[1], reverse=True)[:n]),
            "closeness": dict(sorted(closenesses.items(), key=lambda i: i[1], reverse=True)[:n]),
            "betweenness": dict(sorted(betweennesses.items(), key=lambda i: i[1], reverse=True)[:n])
        }

    return {
        "degree": sorted(degrees, key=degrees.get, reverse=True)[:n],
        "weighted_degree": sorted(weighted_degrees, key=weighted_degrees.get, reverse=True)[:n],
        "closeness": sorted(closenesses, key=closenesses.get, reverse=True)[:n],
        "betweenness": sorted(betweennesses, key=betweennesses.get, reverse=True)[:n]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The path to the interaction network json file.")
    parser.add_argument("-o", "--output", required=True, help="The path to the statistics json file.")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as file:
        interaction_network = json.load(file)

    # Reconstruct the nx graph
    g = nx.Graph()
    for character1, interactions in interaction_network.items():
        for character2, weight in interactions.items():
            g.add_edge(character1, character2, weight=weight)

    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    top_stats = compute_top_centralities(g, 10, with_values=True)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(top_stats, file, indent=4)


if __name__ == '__main__':
    main()
