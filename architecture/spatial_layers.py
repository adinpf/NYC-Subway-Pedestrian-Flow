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

    def call(self, inputs: tuple):
        '''
        inputs: tuple(x,a) where x=node features, a=adjacency
        '''
        x, a = inputs    
        for layer in self.layers_list:
            # GCNConv input order is [X, A]
            if isinstance(layer, GCNConv):
                x = layer([x, a])
            else:
                x = layer(x)
        return x
    
class stackedSpatialGCNs(keras.layers.Layer):
    def __init__(self, *blocks, **kwargs):
        '''
        blocks = sequence of layers that each take (g, h) and return h
        add residual connections between each block
        '''
        super().__init__(**kwargs)
        self.blocks = list(blocks)

    def call(self, inputs, training=False):
        g, h = inputs
        # apply all but last with residual
        for block in self.blocks[:-1]:
            h = h + block((g, h), training=training)
        # apply the last one without residual
        h = self.blocks[-1]((g, h), training=training)
        return h

