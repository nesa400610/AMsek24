import networkx as nx
import torch
from torch_geometric.data import Data

def create_stock_graph(data):
    """
    Create a graph from stock data correlation matrix.
    """
    corr_matrix = data.corr()
    graph = nx.from_pandas_adjacency(corr_matrix)
    return graph

def prepare_data_for_gnn(graph, data):
    """
    Prepare data for Graph Neural Network.
    """
    node_features = torch.tensor(data.values, dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges())).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)
