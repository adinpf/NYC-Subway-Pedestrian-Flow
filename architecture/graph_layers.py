from spektral.layers.convolutional import GCNConv
import tensorflow as tf
import keras
from keras.layers import BatchNormalization, ReLU

class GCN(keras.Model):
    def __init__(self, in_feats, hidden_sizes, out_feats):
        '''
        gcns with hidden layers

        params:
            hidden_sizes: array of sizes for hidden layers
            out_feats: number of output features
        '''
        super().__init__()
        self.layers_list = []
        # stack hidden GCN layers
        for h in hidden_sizes:
            self.layers_list += [
                GCNConv(h, activation=None),
                BatchNormalization(),
                ReLU(),
            ]
        # final output gcn 
        self.layers_list += [
            GCNConv(out_feats, activation=None),
            BatchNormalization(),
            ReLU(),
        ]

    def call(self, inputs):
        x, a = inputs    # x: node features, a: adjacency (dense or sparse)
        for layer in self.layers_list:
            # GCNConv takes a list [X, A]
            if isinstance(layer, GCNConv):
                x = layer([x, a])
            else:
                x = layer(x)
        return x
