"""
ASNCD: Adaptive Scalable Node Community Detection

A distributed community detection algorithm for large-scale graphs.
"""

import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
import os
import time
import leidenalg as la
import igraph as ig
from collections import defaultdict, Counter, deque
import random
import multiprocessing as mp
from functools import partial
import warnings
from sklearn.exceptions import ConvergenceWarning
import metis
import math
import torch

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Hyperparameters
HIDDEN_DIM = 16
EPOCHS = 20
NUM_PROCESSES = mp.cpu_count()
LEIDEN_ITERATIONS = 20
TEMP_DIR = "temp_embeddings"
RANDOM_SEED = 42

# Fix random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def enforce_array_type(arr, dtype=np.float32, shape=None):
    """Ensure array has correct type and shape"""
    try:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=dtype)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        if shape is not None and arr.shape != shape:
            arr = np.zeros(shape, dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            arr[np.isnan(arr) | np.isinf(arr)] = 0.0
        return arr
    except Exception:
        return np.zeros(shape, dtype=dtype) if shape else np.array([], dtype=dtype)


def get_adaptive_params(N, E):
    """
    Adaptive parameter calculation based on graph properties.
    
    Returns:
        block_size: Target block size for partitioning
        tau: Minimum community size threshold
        alpha: Common neighbor weight factor
    """
    BYTES_PER_NODE = 100
    TARGET_MEM_MB = 50
    BLOCKS_PER_CORE = 3
    num_workers = 5
    
    SPARSE_DEGREE = 1
    DENSE_DEGREE = 30
    SPARSE_FACTOR = 0
    DENSE_FACTOR = 2
    
    avg_deg = 2 * E / N
    
    if avg_deg < SPARSE_DEGREE:
        density = SPARSE_FACTOR
    elif avg_deg > DENSE_DEGREE:
        density = DENSE_FACTOR
    else:
        density = SPARSE_FACTOR + (DENSE_FACTOR - SPARSE_FACTOR) / (DENSE_DEGREE - SPARSE_DEGREE) * (avg_deg - SPARSE_DEGREE)
    
    max_by_mem = (TARGET_MEM_MB * 1024 * 1024) // BYTES_PER_NODE
    max_by_parallel = N // (num_workers * BLOCKS_PER_CORE)
    block = int(min(max_by_mem, max_by_parallel) * density)
    block_size = max(1000, min(block, 50000))
    
    d_critical = 6
    base_tau = max(2, int(math.log10(N)))
    
    if avg_deg < d_critical:
        tau = max(2, base_tau - 1)
    else:
        tau = base_tau
    
    ALPHA_SPARSE = 2
    ALPHA_DENSE = 0
    
    if avg_deg <= SPARSE_DEGREE:
        alpha = ALPHA_SPARSE
    elif avg_deg >= DENSE_DEGREE:
        alpha = ALPHA_DENSE
    else:
        slope = (ALPHA_DENSE - ALPHA_SPARSE) / (DENSE_DEGREE - SPARSE_DEGREE)
        alpha = ALPHA_SPARSE + slope * (avg_deg - SPARSE_DEGREE)
    
    return block_size, tau, alpha


def split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size):
    """Split data based on graph connectivity using BFS"""
    visited = set()
    adjacency = defaultdict(list)
    
    for u, v in edge_df[['u', 'v']].values:
        adjacency[u].append(v)
        adjacency[v].append(u)
    
    sorted_nodes = sorted(all_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
    
    blocks = []
    current_block = []
    
    for node in sorted_nodes:
        if node in visited:
            continue
        
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            current_block.append(current)
            
            if len(current_block) >= block_size:
                blocks.append(current_block)
                current_block = []
            
            neighbors = adjacency.get(current, [])
            neighbors_sorted = sorted(neighbors, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
            for neighbor in neighbors_sorted:
                queue.append(neighbor)
    
    if current_block:
        if len(current_block) <= block_size and len(blocks) > 0:
            if len(blocks[-1]) + len(current_block) <= block_size * 1.5:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        else:
            blocks.append(current_block)
    
    unvisited_nodes = [node for node in all_nodes if node not in visited]
    for i in range(0, len(unvisited_nodes), block_size):
        small_block = unvisited_nodes[i:i+block_size]
        if small_block:
            blocks.append(small_block)
    
    new_all_nodes = []
    for block in blocks:
        new_all_nodes.extend(block)
    new_all_nodes = list(set(new_all_nodes))
    
    return blocks, new_all_nodes


def split_data_random(all_nodes, block_size):
    """Random partitioning (baseline comparison)"""
    nodes_shuffled = list(all_nodes)
    random.shuffle(nodes_shuffled)
    
    blocks = []
    for i in range(0, len(nodes_shuffled), block_size):
        block = nodes_shuffled[i:i+block_size]
        if block:
            blocks.append(block)
    
    return blocks


def split_data_by_metis(edge_df, all_nodes, block_size):
    """Use Metis for structure-aware partitioning"""
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    
    num_blocks = len(all_nodes) // block_size
    edges = []
    
    for u, v in edge_df[['u', 'v']].values:
        G.add_edge(u, v)
        edges.append((u, v))
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    adj_list = []
    for node in all_nodes:
        neighbors = [node_to_idx[n] for n in G.neighbors(node)]
        adj_list.append(sorted(neighbors))
    
    try:
        edgecuts, parts = metis.part_graph(adj_list, nparts=num_blocks)
    except TypeError:
        edgecuts, parts = metis.part_graph(adj_list, nparts=num_blocks, options=[5, 0, 0])
    
    blocks_dict = defaultdict(list)
    for i, part in enumerate(parts):
        node = idx_to_node[i]
        blocks_dict[part].append(node)
    
    blocks = list(blocks_dict.values())
    
    cut_edges = 0
    for u, v in edges:
        if parts[node_to_idx[u]] != parts[node_to_idx[v]]:
            cut_edges += 1
    
    return blocks, cut_edges


def split_data_by_spectral(edge_df, all_nodes, block_size):
    """Use spectral clustering as METIS alternative"""
    from scipy.sparse import csr_matrix
    from sklearn.cluster import SpectralClustering
    
    num_blocks = len(all_nodes) // block_size
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    rows, cols = [], []
    for u, v in edge_df[['u', 'v']].values:
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
    
    data = np.ones(len(rows))
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(len(all_nodes), len(all_nodes)))
    
    sc = SpectralClustering(
        n_clusters=num_blocks,
        affinity='precomputed',
        random_state=42,
        n_init=10,
        assign_labels='kmeans'
    )
    labels = sc.fit_predict(adj_matrix)
    
    blocks_dict = defaultdict(list)
    for i, label in enumerate(labels):
        blocks_dict[label].append(all_nodes[i])
    
    cut_edges = 0
    for u, v in edge_df[['u', 'v']].values:
        if labels[node_to_idx[u]] != labels[node_to_idx[v]]:
            cut_edges += 1
    
    return list(blocks_dict.values()), cut_edges


def split_data_by_node_strict(all_nodes, node_degree_dict, block_size):
    """Strict sequential partitioning by node ID"""
    start_time = time.time()
    
    unique_nodes = sorted(set(all_nodes))
    total_nodes = len(unique_nodes)
    
    blocks = []
    for i in range(0, total_nodes, block_size):
        end_idx = min(i + block_size, total_nodes)
        block_nodes = unique_nodes[i:end_idx]
        blocks.append(block_nodes)
    
    elapsed_time = time.time() - start_time
    print(f"Data partitioning (strict order): {len(blocks)} blocks, {total_nodes} nodes, {elapsed_time:.4f}s")
    
    return blocks, unique_nodes


def split_data_by_degree(all_nodes, node_degree_dict, block_size, descending=True):
    """Partition by node degree"""
    start_time = time.time()
    
    unique_nodes = list(set(all_nodes))
    sorted_nodes = sorted(unique_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=descending)
    total_nodes = len(sorted_nodes)
    
    blocks = [sorted_nodes[i:i+block_size] for i in range(0, total_nodes, block_size)]
    
    elapsed_time = time.time() - start_time
    order = "descending (high-degree first)" if descending else "ascending (low-degree first)"
    print(f"Data partitioning (by degree {order}): {len(blocks)} blocks, {elapsed_time:.4f}s")
    
    return blocks, unique_nodes


def calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha, 
                                    node_embed_dict=None, embed_weight_alpha=0.3):
    """Calculate edge weights for a block with optional embedding-based adjustment"""
    try:
        start_time = time.time()
        block_node_set = set(block_nodes)
        
        block_mask = edge_df['u'].isin(block_node_set) & edge_df['v'].isin(block_node_set)
        block_edge = edge_df[block_mask].copy()
        edge_count = len(block_edge)
        
        if edge_count == 0:
            return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))
        
        block_edge[['u_sorted', 'v_sorted']] = np.sort(block_edge[['u', 'v']].values, axis=1)
        edge_counts = block_edge.groupby(['u_sorted', 'v_sorted']).size().reset_index(name='count')
        
        neighbor_dict = defaultdict(list)
        for _, row in edge_counts.iterrows():
            u, v = row['u_sorted'], row['v_sorted']
            neighbor_dict[u].append(v)
            neighbor_dict[v].append(u)
        
        for u in neighbor_dict:
            neighbor_dict[u].sort()
        
        def count_common(u, v):
            neighbors_u = neighbor_dict.get(u, [])
            neighbors_v = neighbor_dict.get(v, [])
            i = j = common = 0
            len_u, len_v = len(neighbors_u), len(neighbors_v)
            
            while i < len_u and j < len_v:
                if neighbors_u[i] == neighbors_v[j]:
                    common += 1
                    i += 1
                    j += 1
                elif neighbors_u[i] < neighbors_v[j]:
                    i += 1
                else:
                    j += 1
            return common
        
        edge_counts['common'] = edge_counts.apply(
            lambda row: count_common(row['u_sorted'], row['v_sorted']), axis=1
        )
        
        edge_counts['weight'] = edge_counts['count'] + cn_base_alpha * edge_counts['common']
        
        result_edge = block_edge[['u', 'v']].drop_duplicates()
        result_edge = result_edge.merge(
            edge_counts[['u_sorted', 'v_sorted', 'weight']],
            left_on=['u', 'v'],
            right_on=['u_sorted', 'v_sorted'],
            how='left'
        ).fillna(1)[['u', 'v', 'weight']]
        
        result_edge['weight'] = result_edge['weight'].clip(lower=0.001)
        
        return (block_id, result_edge)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))


def simplest_structural_embedding(edges, nodes, block_id):
    """Simple structural embedding based on graph structure"""
    G = nx.Graph()
    for idx, row in edges.iterrows():
        G.add_edge(row['u'], row['v'])
    
    for node in nodes:
        if node not in G:
            G.add_node(node)
    
    embeddings = {}
    for node in nodes:
        deg = G.degree(node)
        clustering = nx.clustering(G, node)
        
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            avg_neighbor_deg = sum(G.degree(n) for n in neighbors) / len(neighbors)
        else:
            avg_neighbor_deg = 0
        
        vec = np.array([deg, clustering, avg_neighbor_deg], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings[node] = vec
    
    return (block_id, embeddings)


def minimal_community_aware_embedding(edges, nodes, block_id):
    """Optimized community-aware embedding"""
    degree = {}
    neighbor_counts = {}
    
    for _, row in edges.iterrows():
        u, v = row['u'], row['v']
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        
        if u not in neighbor_counts:
            neighbor_counts[u] = set()
        if v not in neighbor_counts:
            neighbor_counts[v] = set()
        
        neighbor_counts[u].add(v)
        neighbor_counts[v].add(u)
    
    n_nodes = len(nodes)
    embeddings = {}
    
    max_degree = max(degree.values()) if degree else 1
    avg_degree = sum(degree.values()) / n_nodes if n_nodes > 0 else 0
    
    for i, node in enumerate(nodes):
        deg = degree.get(node, 0)
        
        if node in neighbor_counts:
            neighbors = neighbor_counts[node]
            neighbor_degrees = [degree.get(n, 0) for n in neighbors]
            avg_neighbor_deg = np.mean(neighbor_degrees) if neighbor_degrees else 0
            
            neighbor_connections = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and n2 in neighbor_counts.get(n1, set()):
                        neighbor_connections += 1
            
            possible_connections = len(neighbors) * (len(neighbors) - 1) if len(neighbors) > 1 else 1
            clustering = neighbor_connections / possible_connections if possible_connections > 0 else 0
            boundary_strength = deg * (1 - clustering) if clustering < 1 else 0
        else:
            avg_neighbor_deg = 0
            clustering = 0
            boundary_strength = 0
        
        vec = np.zeros(8, dtype=np.float32)
        vec[0] = deg
        vec[1] = deg / max_degree if max_degree > 0 else 0
        vec[2] = 1 if deg > avg_degree else -1 if deg < avg_degree else 0
        vec[3] = boundary_strength / max_degree if max_degree > 0 else 0
        vec[4] = clustering
        vec[5] = avg_neighbor_deg / max_degree if max_degree > 0 else 0
        
        node_hash = hash(str(node)) % 10000
        vec[6] = np.sin(node_hash / 1000.0)
        vec[7] = np.cos(node_hash / 777.0)
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        embeddings[node] = vec
    
    return (block_id, embeddings)


def generate_block_community(args):
    """Generate communities for a block using Leiden algorithm"""
    try:
        weighted_edge_block, block_nodes, block_id = args
        start_time = time.time()
        
        block_G = nx.Graph()
        if not weighted_edge_block.empty:
            valid_edges = weighted_edge_block[['u', 'v', 'weight']].values
            if len(valid_edges) > 0:
                block_G.add_weighted_edges_from(valid_edges)
        
        for node in block_nodes:
            if node not in block_G:
                block_G.add_node(node)
        
        if not block_G.nodes():
            return (block_id, {})
        
        node_list = sorted(block_G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        block_ig = ig.Graph(directed=False)
        block_ig.add_vertices(len(node_list))
        
        edges, edge_weights = [], []
        for u, v, data in block_G.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            edge_weights.append(float(data.get('weight', 1.0)))
        
        if edges:
            block_ig.add_edges(edges)
            block_ig.es['weight'] = edge_weights
        
        partition = la.find_partition(
            block_ig,
            la.ModularityVertexPartition,
            weights='weight',
            n_iterations=20,
            seed=42
        )
        
        leiden_comm = np.array(partition.membership)
        block_fine_comm = [leiden_comm[node_to_idx[node]] if node in node_to_idx else 0 
                          for node in block_nodes]
        
        global_comm_prefix = block_id * 1000000
        block_comm_dict = {node: global_comm_prefix + comm 
                          for node, comm in zip(block_nodes, block_fine_comm)}
        
        return (block_id, block_comm_dict)
    
    except Exception as e:
        block_comm_dict = {node: block_id * 1000000 + i for i, node in enumerate(block_nodes)}
        return (block_id, block_comm_dict)


def build_global_graph_from_original_optimized(edge_df, nodes):
    """Build global graph from original edges (optimized version)"""
    start_time = time.time()
    
    if isinstance(edge_df, pd.DataFrame):
        edges_array = edge_df[['u', 'v']].values
    else:
        edges_array = edge_df
    
    edges_sorted = np.sort(edges_array, axis=1)
    
    edge_counts = {}
    batch_size = 1000000
    
    for i in range(0, len(edges_sorted), batch_size):
        batch = edges_sorted[i:i+batch_size]
        for u, v in batch:
            key = (u, v)
            edge_counts[key] = edge_counts.get(key, 0) + 1.0
    
    edge_list = [(u, v, w) for (u, v), w in edge_counts.items()]
    
    G = nx.Graph()
    
    if nodes is not None:
        G.add_nodes_from(nodes)
    else:
        node_set = set()
        for (u, v), _ in edge_counts.items():
            node_set.add(u)
            node_set.add(v)
        G.add_nodes_from(node_set)
    
    G.add_weighted_edges_from(edge_list)
    
    return G


def build_lightweight_graph(edge_df, nodes):
    """Build lightweight graph structure as NetworkX alternative"""
    if nodes is not None:
        all_nodes = list(nodes)
    else:
        if isinstance(edge_df, pd.DataFrame):
            edges_array = edge_df[['u', 'v']].values
        else:
            edges_array = edge_df
        all_nodes = list(set(edges_array.flatten()))
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    adjacency = defaultdict(list)
    
    if isinstance(edge_df, pd.DataFrame):
        edges_array = edge_df[['u', 'v']].values
    else:
        edges_array = edge_df
    
    edge_weights = defaultdict(float)
    for u, v in edges_array:
        i, j = node_to_idx[u], node_to_idx[v]
        if i <= j:
            edge_weights[(i, j)] += 1.0
    
    for (i, j), w in edge_weights.items():
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))
    
    class LightweightGraph:
        def __init__(self, adjacency, idx_to_node, node_to_idx):
            self.adj = adjacency
            self.idx_to_node = idx_to_node
            self.node_to_idx = node_to_idx
            self.edge_weights = edge_weights
            self._nodes = set(idx_to_node.values())
        
        def __contains__(self, node):
            return node in self.node_to_idx
        
        def neighbors(self, node):
            idx = self.node_to_idx[node]
            return [self.idx_to_node[neighbor_idx] for neighbor_idx, _ in self.adj[idx]]
        
        def has_node(self, node):
            return node in self.node_to_idx
        
        def nodes(self):
            return self._nodes
        
        def edges(self, data=False):
            if data:
                return [(self.idx_to_node[i], self.idx_to_node[j], {'weight': w}) 
                        for (i, j), w in self.edge_weights.items()]
            else:
                return [(self.idx_to_node[i], self.idx_to_node[j]) 
                        for (i, j) in self.edge_weights.keys()]
    
    return LightweightGraph(adjacency, idx_to_node, node_to_idx)


def merge_small_communities_fast(partition_dict, edge_df, min_size=3):
    """Fast merging of small communities"""
    communities = partition_dict.copy()
    
    small_comms = {}
    large_comms = {}
    node_to_comm = {}
    
    for cid, nodes in communities.items():
        node_list = list(nodes)
        if len(node_list) < min_size:
            small_comms[cid] = node_list
        else:
            large_comms[cid] = node_list
        for node in node_list:
            node_to_comm[node] = cid
    
    if not small_comms:
        return communities
    
    comm_connections = defaultdict(lambda: defaultdict(int))
    for u, v in edge_df.values:
        comm_u = node_to_comm[u]
        comm_v = node_to_comm[v]
        
        if comm_u != comm_v:
            comm_connections[comm_u][comm_v] += 1
            comm_connections[comm_v][comm_u] += 1
    
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}
    
    for small_cid, small_nodes in small_comms.items():
        connections = comm_connections.get(small_cid, {})
        
        candidate_large = {}
        for neighbor_comm, weight in connections.items():
            if neighbor_comm in merged_result:
                candidate_large[neighbor_comm] = weight
        
        if candidate_large:
            best_comm = max(
                candidate_large.items(),
                key=lambda x: (x[1], comm_size.get(x[0], 0))
            )[0]
        else:
            if comm_size:
                best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
            else:
                best_comm = max(merged_result.keys(), default=-1) + 1
                merged_result[best_comm] = set()
                comm_size[best_comm] = 0
        
        if best_comm not in merged_result:
            merged_result[best_comm] = set()
            comm_size[best_comm] = 0
        
        merged_result[best_comm].update(small_nodes)
        comm_size[best_comm] += len(small_nodes)
    
    final_result = {}
    for new_id, (old_id, nodes) in enumerate(merged_result.items()):
        if nodes:
            final_result[new_id] = list(nodes)
    
    sizes = [len(nodes) for nodes in final_result.values()]
    remaining_small = sum(1 for size in sizes if size < min_size)
    
    if sizes:
        print(f"Community sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
    
    original_nodes = sum(len(nodes) for nodes in communities.values())
    result_nodes = sum(len(nodes) for nodes in final_result.values())
    assert original_nodes == result_nodes, f"Node count mismatch: {original_nodes} != {result_nodes}"
    
    return final_result


def optimize_community_structure(node_to_community_dict, edge_df=None, min_size=3):
    """Optimize community structure"""
    print("Optimizing community structure...")
    
    community_to_nodes = defaultdict(list)
    for node, comm_id in node_to_community_dict.items():
        community_to_nodes[comm_id].append(node)
    
    merged_community_to_nodes = merge_small_communities_fast(community_to_nodes, edge_df, min_size)
    
    final_node_to_community = {}
    for comm_id, nodes in merged_community_to_nodes.items():
        for node in nodes:
            final_node_to_community[node] = comm_id
    
    return final_node_to_community


def global_optimization_with_overlap(G, comm_dict, all_new_nodes):
    """Global optimization with overlap nodes"""
    improved_comm_dict = comm_dict.copy()
    comm_sizes = defaultdict(int)
    
    for node, comm_id in improved_comm_dict.items():
        comm_sizes[comm_id] += 1
    
    for new_node in all_new_nodes:
        if new_node not in G:
            continue
        
        current_comm = improved_comm_dict[new_node]
        neighbors = list(G.neighbors(new_node))
        
        if not neighbors:
            continue
        
        neighbor_comms = defaultdict(int)
        for neighbor in neighbors:
            neighbor_comm = improved_comm_dict.get(neighbor, -1)
            if neighbor_comm != -1:
                neighbor_comms[neighbor_comm] += 1
        
        if neighbor_comms:
            best_comm = max(
                neighbor_comms.items(),
                key=lambda x: (x[1], comm_sizes.get(x[0], 0))
            )[0]
            
            current_conn = neighbor_comms.get(current_comm, 0)
            best_conn = neighbor_comms[best_comm]
            
            if best_comm != current_comm and best_conn > current_conn:
                comm_sizes[current_comm] = max(0, comm_sizes.get(current_comm, 0) - 1)
                comm_sizes[best_comm] = comm_sizes.get(best_comm, 0) + 1
                improved_comm_dict[new_node] = best_comm
    
    return improved_comm_dict


def convert_comm_dict_to_comms_list(comm_dict):
    """Convert node->community dict to list of community lists"""
    comm_to_nodes = defaultdict(list)
    for node, comm_id in comm_dict.items():
        comm_to_nodes[comm_id].append(node)
    
    return [nodes for nodes in comm_to_nodes.values()]


def evaluate_with_correct_format(true_comms, comm_dict):
    """Evaluate with correct format"""
    pred_comms = convert_comm_dict_to_comms_list(comm_dict)
    
    true_sizes = [len(comm) for comm in true_comms]
    pred_sizes = [len(comm) for comm in pred_comms]
    
    try:
        from metrics import eval_scores_fast_optimized_fixed
        avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(
            pred_comms, true_comms, tmp_print=True
        )
        
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        print(f"  Average Jaccard: {avg_jaccard:.4f}")
        
        return avg_precision, avg_recall, avg_f1, avg_jaccard
    except ImportError:
        return 0.0, 0.0, 0.0, 0.0


def load_data(edge_file_path, comm_file_path):
    """Load SNAP Community datasets"""
    with open(comm_file_path) as f:
        communities = [[int(i) for i in x.split()] for x in f]
    
    with open(edge_file_path) as f:
        edges = [[int(i) for i in e.split()] for e in f]
    
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    
    raw_nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(raw_nodes))}
    
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communities = [[mapping[node] for node in com] for com in communities]
    
    num_node = len(raw_nodes)
    num_edges = len(edges)
    num_comm = len(communities)
    
    print(f"[{os.path.basename(edge_file_path).upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")
    
    new_nodes = list(range(len(raw_nodes)))
    
    return num_node, num_edges, num_comm, new_nodes, edges, communities


def compute_degree_cv(edges, total_nodes, sample_size=10000):
    """Calculate degree distribution coefficient of variation"""
    if total_nodes < 100000:
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        G.add_edges_from(edges)
        degrees = [d for n, d in G.degree()]
    else:
        degree_count = Counter()
        for u, v in edges:
            degree_count[u] += 1
            degree_count[v] += 1
        
        sampled_nodes = np.random.choice(
            list(degree_count.keys()),
            size=min(sample_size, len(degree_count)),
            replace=False
        )
        degrees = [degree_count[n] for n in sampled_nodes]
    
    avg_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    cv_degree = std_deg / avg_deg if avg_deg > 0 else 0
    
    return cv_degree


def choose_algorithm(n_nodes, avg_deg, cv_deg):
    """Choose algorithm based on graph properties"""
    if avg_deg <= 10 and n_nodes > 500_000 and cv_deg > 5:
        return 2
    if avg_deg <= 10 and n_nodes <= 200_000:
        return 2
    if avg_deg <= 10 and n_nodes > 200_000:
        return 1
    if avg_deg > 20 and n_nodes < 100_000 and cv_deg > 1.5:
        return 2
    if avg_deg > 20:
        return 1
    return 1


def execute_ASNCD_pipeline_unsupervised(edge_file_path, comm_file_path, network_type):
    """Main unsupervised ASNCD pipeline"""
    start_total_time1 = time.time()
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    num_node, num_edges, num_comm, all_nodes, edges, communities = load_data(
        edge_file_path, comm_file_path
    )
    edge_df = pd.DataFrame(edges, columns=['u', 'v'])
    
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    u_idx = np.array([node_to_idx.get(u, -1) for u in edge_df['u']], dtype=np.int32)
    v_idx = np.array([node_to_idx.get(v, -1) for v in edge_df['v']], dtype=np.int32)
    valid_u = u_idx[u_idx != -1]
    valid_v = v_idx[v_idx != -1]
    u_degrees = np.bincount(valid_u, minlength=len(all_nodes))
    v_degrees = np.bincount(valid_v, minlength=len(all_nodes))
    node_degree = (u_degrees + v_degrees).astype(np.float32)
    node_degree_dict = dict(zip(all_nodes, node_degree))
    
    total_time1 = (time.time() - start_total_time1) / 60
    print(f"\nTotal time: {total_time1:.2f} minutes")
    
    block_size, MIN_COMM_SIZE, cn_base_alpha = get_adaptive_params(len(all_nodes), len(edge_df))
    
    print(f"  block_size: {block_size}, MIN_COMM_SIZE: {MIN_COMM_SIZE}, cn_base_alpha: {cn_base_alpha}")
    
    klist = [block_size]
    
    for block_size in klist:
        start_total_time = time.time()
        
        print(f"  block_size: {block_size}, MIN_COMM_SIZE: {MIN_COMM_SIZE}, cn_base_alpha: {cn_base_alpha}")
        print(f"Block size: {block_size}")
        start_cpp = time.time()
        
        blocks, new_all_nodes = split_data_by_connectivity(
            edge_df, all_nodes, node_degree_dict, block_size
        )
        
        end_cpp = (time.time() - start_cpp) / 60
        print(f"\nBFS partitioning time: {end_cpp:.2f} minutes")
        print(f"Generated {len(blocks)} blocks")
        
        num_blocks = len(blocks)
        
        if num_blocks == 0:
            raise ValueError("Data splitting failed")
        
        with Pool(processes=NUM_PROCESSES) as pool:
            block_args = [(block_id, block_nodes) for block_id, block_nodes in enumerate(blocks)]
            partial_func = partial(process_block, edge_df=edge_df, cn_base_alpha=cn_base_alpha)
            results = pool.imap_unordered(partial_func, block_args)
            
            weighted_edge_dict = {}
            for bid, bedge in results:
                weighted_edge_dict[bid] = bedge
        
        weighted_edge_list = []
        for i in sorted(weighted_edge_dict.keys()):
            df = weighted_edge_dict[i]
            if not df.empty and len(df) > 0:
                weighted_edge_list.append(df)
        
        if weighted_edge_list:
            weighted_edge_df = pd.concat(weighted_edge_list, ignore_index=True)
            weighted_edge_df = weighted_edge_df.astype({
                'u': int,
                'v': int,
                'weight': float
            })
        else:
            weighted_edge_df = pd.DataFrame(columns=['u', 'v', 'weight'])
        
        print(f"Summarized edge weights: {len(weighted_edge_df)} edges")
        
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))
        
        print(f"\n[5/6 Using Leiden for block community detection...]")
        comm_args = []
        for block_id, block_nodes in enumerate(blocks):
            block_edges = weighted_edge_df[
                weighted_edge_df['u'].isin(block_nodes) |
                weighted_edge_df['v'].isin(block_nodes)
            ]
            comm_args.append((block_edges, block_nodes, block_id))
        
        global_comm_dict = {}
        with Pool(processes=NUM_PROCESSES) as pool:
            comm_results = pool.imap_unordered(generate_block_community, comm_args)
            for result in comm_results:
                bid, bcomm = result
                global_comm_dict.update(bcomm)
        
        global_G = build_lightweight_graph(edge_df, all_nodes)
        
        start_new = time.time()
        
        final_comm_dict1 = global_optimization_with_overlap(
            global_G,
            global_comm_dict,
            all_nodes,
        )
        
        end_new = (time.time() - start_new) / 60
        print(f"\nGlobal optimization time: {end_new:.2f} minutes")
        
        final_comm_dict = optimize_community_structure(final_comm_dict1, edge_df, MIN_COMM_SIZE)
        
        if communities:
            metrics = evaluate_with_correct_format(communities, final_comm_dict)
        
        total_time = (time.time() - start_total_time) / 60
        print(f"\nTotal time: {total_time + total_time1:.2f} minutes")
    
    return global_comm_dict


def process_block(args, edge_df, cn_base_alpha):
    """Process block for edge weight calculation"""
    block_id, block_nodes = args
    return calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha)


# Dataset configurations
DATASET_CONFIGS = {
    'facebook': {
        'edge_path': 'dataset/facebook-1.90.ungraph.txt',
        'community_path': 'dataset/facebook-1.90.cmty.txt',
        'description': 'Facebook social network',
        'network_type': 'social'
    },
    'amazon1': {
        'edge_path': 'dataset/amazon-1.90.ungraph.txt',
        'community_path': 'dataset/amazon-1.90.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase'
    },
    'lj1': {
        'edge_path': 'dataset/lj-1.90.ungraph.txt',
        'community_path': 'dataset/lj-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
    'dblp1': {
        'edge_path': 'dataset/dblp-1.90.ungraph.txt',
        'community_path': 'dataset/dblp-1.90.cmty.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration'
    },
    'dblp2': {
        'edge_path': 'dataset/dblp.ungraph.txt',
        'community_path': 'dataset/dblp_communities.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration'
    },
    'amazon2': {
        'edge_path': 'dataset/com-amazon.ungraph.txt',
        'community_path': 'dataset/com-amazon.all.dedup.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase'
    },
    'lj2': {
        'edge_path': 'dataset/lj.ungraph.txt',
        'community_path': 'dataset/lj.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
    'lj3': {
        'edge_path': 'dataset/com-lj.ungraph.txt',
        'community_path': 'dataset/com-lj.all.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
    'youtube': {
        'edge_path': 'dataset/com-youtube.ungraph.txt',
        'community_path': 'dataset/com-youtube.all.cmty.txt',
        'description': 'YouTube social network',
        'network_type': 'social'
    },
   
}


if __name__ == "__main__":
    dslist = ["facebook", "amazon1", "lj1", "dblp1", "amazon2", "lj2", "dblp2"]
    
    for dataset_name in dslist:
        config = DATASET_CONFIGS.get(dataset_name)
        
        EDGE_FILE_PATH = config["edge_path"]
        COMMUNITY_FILE_PATH = config["community_path"]
        network_type = config["network_type"]
        
        if not os.path.exists(EDGE_FILE_PATH):
            raise FileNotFoundError(f"Edge file not found: {EDGE_FILE_PATH}")
        
        if not os.path.exists(COMMUNITY_FILE_PATH):
            print(f"Warning: Community file not found, running unsupervised community detection")
        
        execute_ASNCD_pipeline_unsupervised(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type)
