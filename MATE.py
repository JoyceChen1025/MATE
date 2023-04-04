import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import MessageAggregator
from modules.memory_updater import MemoryUpdater
from modules.temporal_embedding import GraphSumEmbedding
from modules.time_encoding import TimeEncode


class MATE(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               use_memory=False, memory_dimension=500, n_neighbors=None):
    super(MATE, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    if self.use_memory:
      self.memory_dimension = memory_dimension
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = MessageAggregator(device=device)
      self.memory_updater = MemoryUpdater(memory=self.memory,
                                          message_dimension=message_dimension,
                                          memory_dimension=self.memory_dimension,
                                          device=device)


    self.embedding_module = GraphSumEmbedding(node_features=node_features,
                                              edge_features=edge_features,
                                              memory=self.memory,
                                              neighbor_finder=neighbor_finder,
                                              time_encoder=self.time_encoder,
                                              n_layers=self.n_layers,
                                              n_node_features=self.n_node_features,
                                              n_edge_features=self.n_edge_features,
                                              n_time_features=self.n_node_features,
                                              embedding_dimension=self.embedding_dimension,
                                              device=self.device,
                                              use_memory=self.use_memory)

    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)
    
    self.mlp_decoder = torch.nn.Linear(self.embedding_dimension, self.n_node_features)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    if self.use_memory:
      memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      self.update_memory(positives, self.memory.messages)

      self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      self.memory.store_raw_messages(unique_sources, source_id_to_messages)
      self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)


    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    pos_score = self.affinity_score(source_node_embedding, destination_node_embedding).view(-1)
    neg_score = self.affinity_score(source_node_embedding, negative_node_embedding).view(-1)

    decode_node = self.mlp_decoder(torch.cat([source_node_embedding, destination_node_embedding], dim=0))
    raw_node_feature = torch.cat([self.node_raw_features[source_nodes], self.node_raw_features[destination_nodes]], dim=0)

    return pos_score.sigmoid(), neg_score.sigmoid(), decode_node, raw_node_feature

  def update_memory(self, nodes, messages):
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = unique_messages

    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = unique_messages

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = source_node_embedding
    destination_memory = destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
