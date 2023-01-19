import torch
import torch.nn as nn
from typing import Optional, List
from torchdrug.models import ProteinCNN, ESM, GearNet
from torchdrug.layers import MLP
import numpy as np
import torch.nn.functional as F



class ProCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 graph_construct:nn.Module,
                 # seq
                 seq_model: nn.Module,
                 # graph
                 graph_model: nn.Module,
                 graph_struct_crop: nn.Module,
                 graph_attr_crop: nn.Module
                 ):

        super(ProCLIP, self).__init__()

        self.seq_model = seq_model
        self.graph_model = graph_model

        self.graph_construct = graph_construct
        self.graph_struct_crop = graph_struct_crop
        self.graph_attr_crop = graph_attr_crop

        self.proj_seq = nn.Parameter(nn.init.xavier_normal_(torch.rand(self.seq_model.output_dim, embed_dim)))
        self.proj_graph = nn.Parameter(nn.init.xavier_normal_(torch.rand(self.graph_model.output_dim, embed_dim)))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = torch.tensor(torch.ones([]) * np.log(1 / 0.07))

    def encode_seq(self, graph, node_feature, normalize: bool = True):
        seq_repr = self.seq_model(graph, node_feature)
        seq_repr = seq_repr["graph_feature"]
        seq_repr = seq_repr @ self.proj_seq
        seq_repr = F.normalize(seq_repr, dim=-1) if normalize else seq_repr
        return seq_repr

    def encode_graph(self, graph, node_feature, normalize: bool = True):
        argument_graph = self.graph_struct_crop(graph)
        argument_graph = self.graph_attr_crop(argument_graph)
        argument_graph_repr = self.graph_model(argument_graph, argument_graph.node_feature.type(torch.float))
        argument_graph_repr = argument_graph_repr["graph_feature"]
        argument_graph_repr = argument_graph_repr @ self.proj_graph
        argument_graph_repr = F.normalize(argument_graph_repr, dim=-1) if normalize else argument_graph
        return argument_graph_repr

    def forward(self, batch):
        graph = self.graph_construct(batch["graph"])
        node_feature = graph.node_feature.type(torch.float)
        seq_repr = self.encode_seq(graph, node_feature, normalize=True)
        arg_graph_repr = self.encode_graph(graph, node_feature, normalize=True)

        return seq_repr, arg_graph_repr, self.logit_scale.exp()

