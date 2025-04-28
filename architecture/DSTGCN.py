import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="DSTGCN")
class DSTGCN(keras.Model):

    def __init__(self, feature_sizes, **kwargs):
        super().__init__(**kwargs)

        spatial_features, st_features, external_features, out_features = feature_sizes
        self.spatial_embedding = keras.layers.Dense(spatial_features, [20], 15)
        self.spatial_gcn = stackedSpatialGCNs([GCN(15, [15, 15, 15], 15),
                                           GCN(15, [15, 15, 15], 15),
                                           GCN(15, [14, 13, 12, 11], 10)])
        self.temporal_embedding = stackedSpatioTemporalGCNs([STBlock(st_features, 4), STBlock(5, 5), STBlock(10, 10)])

        self.temporal_agg = keras.layers.AveragePooling1D(pool_size=24)

        embedding_sizes = [(external_features * (4 - i) + 10 * i) // 4 for i in (1, 4)]
        external_embedding_layers = [
            keras.layers.Dense(external_features, embedding_sizes[0]),
            keras.layers.Dense(embedding_sizes[0], embedding_sizes[1]),
            keras.layers.Dense(embedding_sizes[1], 10),
        ]
        self.external_embedding = keras.Sequential(external_embedding_layers)
        head = [
            keras.activations.relu(),
            keras.layers.Dense(outfeatures)
        ]

    
    def call(self, somthing_inputs):
        return 'TODO'

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_input_stuff, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_input_stuff: TODO TODO TODO
        :return: None
        # """
        # # shuffle features:
        # range_of_indices = np.arange(train_image_features.shape[0])
        # random_indices = tf.random.shuffle(range_of_indices)
        # train_image_features = tf.gather(train_image_features, random_indices)
        # train_captions = tf.gather(train_captions, random_indices)

        # # remove the last token in the window
        # filtered_decoder_captions = train_captions[:, :-1]
        # # remove first token for loss 
        # filtered_loss_captions = train_captions[:, 1:]

        # # batches
        # for s_ind in range(0, len(train_captions), batch_size):
        #     e_ind = s_ind + batch_size
        #     b_image_features = train_image_features[s_ind:e_ind]
        #     bd_captions = filtered_decoder_captions[s_ind:e_ind]
        #     bl_captions = filtered_loss_captions[s_ind:e_ind]

        #     # forward pass
        #     with tf.GradientTape() as tape:
        #         prbs = self.call(b_image_features, bd_captions)
        #         mask = bl_captions != padding_index
        #         b_loss = self.loss_function(prbs, bl_captions, mask)

        #     gradients = tape.gradient(b_loss, self.trainable_variables)
        #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def test(self, test_input_stuff, batch_size=30):
        """
        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_input_stuff: TODO TODO TODO
        :returns: TODO SOME METRIC 
        """
        # num_batches = int(len(test_captions) / batch_size)

        # total_loss = total_seen = total_correct = 0
        # for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

        #     ## Get the current batch of data, making sure to try to predict the next word
        #     start = end - batch_size
        #     batch_image_features = test_image_features[start:end, :]
        #     decoder_input = test_captions[start:end, :-1]
        #     decoder_labels = test_captions[start:end, 1:]

        #     ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
        #     probs = self(batch_image_features, decoder_input)
        #     mask = decoder_labels != padding_index
        #     num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
        #     loss = self.loss_function(probs, decoder_labels, mask)
        #     accuracy = self.accuracy_function(probs, decoder_labels, mask)

        #     ## Compute and report on aggregated statistics
        #     total_loss += loss
        #     total_seen += num_predictions
        #     total_correct += num_predictions * accuracy

        #     avg_loss = float(total_loss / total_seen)
        #     avg_acc = float(total_correct / total_seen)
        #     avg_prp = np.exp(avg_loss)
        #     print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print()        
        # return avg_prp, avg_acc

    # def get_config(self):
    #     base_config = super().get_config()
    #     config = {
    #         "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
    #     }
    #     return {**base_config, **config}

    # @classmethod
    # def from_config(cls, config):
    #     decoder_config = config.pop("decoder")
    #     decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
    #     return cls(decoder, **config)