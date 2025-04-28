import tensorflow as tf

def accuracy_function(prds, labels):
    """

    Computes the batch accuracy

    :param prds:  
    :param labels: 
    :return: something accuracy
    """
    # TODO
    return 'TODO'


def loss_function(prds, labels):
    """
    Calculates the model cross-entropy loss after one forward pass
    Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

    :param prds:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :return: SOMETHING LOSS TODO
    """
    # TODO
    return 'TODO'