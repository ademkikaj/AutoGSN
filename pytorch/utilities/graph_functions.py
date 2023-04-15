import networkx as nx
import itertools

def find_cliques_size_k(graph, k):
    all_cliques = set()
    for clique in nx.find_cliques(graph):
        if len(clique) == k:
            all_cliques.add(tuple(sorted(clique)))
        elif len(clique) > k:
            for mini_clique in itertools.combinations(clique, k):
                all_cliques.add(tuple(sorted(mini_clique)))
    return all_cliques


def calculate_cliques(graph: nx.Graph, clique_range_start, clique_range_end):
    # all_cliques = []
    count = 0
    my_dict = {}
    for i in range(clique_range_start, clique_range_end):
        cliques = find_cliques_size_k(graph, i)
        for node in graph.nodes():
            try:
                my_dict[node].update(
                    {i: len(nx.cliques_containing_node(graph, node, cliques))}
                )
            except:
                my_dict[node] = {
                    i: len(nx.cliques_containing_node(graph, node, cliques))
                }
        # all_cliques.append(cliques)
        count += len(cliques)
    return my_dict


def count_cliques(nr_of_graphs, graphs, clique_range_start, clique_range_end):
    processed = {}
    for i in range(0, nr_of_graphs):
        curr_graph: nx.Graph = graphs[i]
        print("Progress: %d/%d" % (i, nr_of_graphs))
        features = calculate_cliques(curr_graph, clique_range_start, clique_range_end)
        processed[i] = features
    return processed

def count_cycles(cycles, node):
    acc = 0
    for cycle in cycles:
        if node in cycle:
            acc +=1
    return acc

def get_cycles_of_size(nr_of_graphs, graphs):
    processed = {}
    for i in range(0, nr_of_graphs):
        if i==875:
            print("here")
        graph = graphs[i]
        print("Progress: %d/%d" % (i, nr_of_graphs))
        all_cycles = nx.cycle_basis(graph)
        min_cycle = 3
        max_cycle = 10
        my_dict = {}
        if len(all_cycles) > 0:
            for size in range(min_cycle, max_cycle+1):            
                cycles_of_size = []
                for cycle in all_cycles:
                    if len(cycle) == size:
                        cycles_of_size.append(cycle)
                for node in graph.nodes():
                    try:
                        my_dict[node].update({size: count_cycles(cycles_of_size, node)})
                    except:
                        my_dict[node] = {size: count_cycles(cycles_of_size, node)}
            processed[i] = my_dict
        else:
            for size in range(min_cycle, max_cycle+1):            
                for node in graph.nodes():
                    try:
                        my_dict[node].update({size: 0})
                    except:
                        my_dict[node] = {size: 0}
            processed[i] = my_dict
    return processed
        