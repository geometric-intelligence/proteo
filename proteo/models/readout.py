import torch.nn as nn
import torch


class Readout(nn.Module):
    ACT_MAP = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    def __init__(
            self, 
            feature_output_dim,
            which_layer,
            fc_dim=None,
            fc_dropout=None,
            fc_act=None,
            out_channels=None,
            fc_input_dim=None,
            use_feature_encoder=True
        ):
        super().__init__()
        self.feature_output_dim = feature_output_dim #how big do you want sex, mutation, age to be encoded to
        self.which_layer = which_layer
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
        self.fc_act = fc_act
        self.fc_input_dim = fc_input_dim #how big are graph features + sex, mutation, age
        self.out_channels = out_channels #1
        self.use_feature_encoder = use_feature_encoder
        self.readout_layers = self.build_readout_layers()
        self.feature_encoder = self.build_feature_encoder()

    
    def build_readout_layers(self):
        layers = []
        fc_layer_input_dim = self.fc_input_dim
        for fc_dim in self.fc_dim:
            layers.append(
                nn.Sequential(
                    nn.Linear(fc_layer_input_dim, fc_dim),
                    self.ACT_MAP[self.fc_act],
                    nn.AlphaDropout(p=self.fc_dropout, inplace=True),
                )
            )
            fc_layer_input_dim = fc_dim
        layers.append(nn.Linear(fc_dim, self.out_channels))
        return nn.Sequential(*layers)
    
    def build_feature_encoder(self):
        if self.use_feature_encoder:
            return nn.Linear(1, self.feature_output_dim)
        else:
            return nn.Identity()
    
    def encode_features(self, data):
        sex = data.sex
        mutation = data.mutation
        age = data.age
        encoded_features = []
        for feature in self.which_layer:
            if feature in ['sex','mutation','age']:
                feature_value = locals().get(
                    feature
                ) 
                encoded_features.append(self.feature_encoder(feature_value))
        return torch.cat(encoded_features, dim=1)
    
    def concatenate_features(self, graph_features, demographic_features):
        return torch.cat([graph_features, demographic_features], dim=1)
    
    def forward(self, graph_features, batch):
        demographic_features = self.encode_features(batch)
        total_features = self.concatenate_features(graph_features, demographic_features)
        return self.readout_layers(total_features)