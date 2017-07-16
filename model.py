import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicRNNCell, LSTMCell, static_rnn, BasicLSTMCell
from input_reader import Clues5Generator
from rnn_cells import MyGRUCell, MyLSTMCell, SkipZerosWrapper
import numpy as np
import datetime

batch_size = 64
num_batches_per_epoch = int(1431406/batch_size)
num_batches = 2*num_batches_per_epoch

answer_length = 5
num_chars = 26
encoder_hidden_dim = 75
decoder_hidden_dim = 75

pickled_index_filename = "picked-index.bin"

embedding_dimension = 300 # this is fixed by the word2vec model
max_clue_length = 40 # determined by the input data

def decoder(context, batch_size, output_length, hidden_dim, output_dim):
    """Context - tensor to decode from (shape: (b, ?))"""

    input_dim = context.shape[-1] + output_dim
    
    # could experiment with using context as part of the state, thus having it get updated each time
    # for now, it's part of the input

    cell = MyLSTMCell(batch_size, hidden_dim, input_dim, output_dim)

    state = cell.zero_state()
    output = cell.zero_output()

    answers = tf.zeros( (batch_size, 0, output_dim), dtype=tf.float32)

    for i in range(output_length):
        combined_input = tf.concat([context, output], 1)
        output, state = cell(combined_input, state)
        answers = tf.concat([answers, tf.reshape(output, (batch_size,1,-1))], 1)

    return answers

def encoder(clues, lengths, batch_size, max_length, hidden_dim):

    split_clues = tf.split(clues, max_length, 1)
    split_clues = list(map(lambda x: tf.reshape(x, (batch_size, embedding_dimension)), split_clues))


    with tf.variable_scope("inner_encoder"):
        cell = SkipZerosWrapper(MyLSTMCell(batch_size, hidden_dim, embedding_dimension, 0))

        state = cell.zero_state()

        for i in range(max_length):
            _, state = cell(split_clues[i], state)


    return state[0]


def loss(decoded, answers):
    # both are (batch, answer, character)
    return tf.reduce_sum(tf.squared_difference(decoded, answers))

def all_together_now(clues, lengths, answers, batch_size, max_clue_length, answer_length, encoder_hidden_dim, decoder_hidden_dim, num_chars=26):

    with tf.variable_scope("encoder"):
        enc = encoder(clues, lengths, batch_size, max_clue_length, encoder_hidden_dim)

    with tf.variable_scope("decoder"):
        dec = decoder(enc, batch_size, answer_length, decoder_hidden_dim, num_chars)

    with tf.variable_scope("loss"):
        l   =    loss(dec, answers)

    return enc, dec, l

def convert_letters(arr):
    # arr is np array with shape (n, l)
    # convert into a list of n strings, l letters each
    py = arr.tolist()
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def single_list_to_string(l):
        return ''.join(abc[x] for x in l)
    return map(single_list_to_string, py)

if __name__ == "__main__":
    #instantiate model

    clues = tf.placeholder( tf.float32,
                            shape=(batch_size, max_clue_length, embedding_dimension),
                            name="input_sentence")
    
    lengths = tf.placeholder( tf.int32,
                              (batch_size, ),
                              name="input_length")

    answers = tf.placeholder( tf.float32,
                              (batch_size, answer_length, num_chars),
                              name="answer")

    enc, dec, loss = all_together_now(clues, lengths, answers, batch_size, max_clue_length, answer_length,
                                      encoder_hidden_dim, decoder_hidden_dim)
    top_letters = tf.argmax(dec, axis=-1)


    with tf.variable_scope("train"):
        opt = tf.train.AdamOptimizer()
        train = opt.minimize(loss)

        saver = tf.train.Saver()

        c = Clues5Generator(batch_size, pickled_index=pickled_index_filename, read_pickle=True)

    init_op = tf.global_variables_initializer()


    sess = tf.Session()
    sess.run(init_op)

    start = datetime.datetime.now()

    for i in range(num_train):
        if i % 1000 == 0: print(i)

        x,y,l = c.__next__()
        feed_dict = {answers:y,clues:x,lengths:l}
        sess.run(train, feed_dict=feed_dict)
        
    end = datetime.datetime.now()
    delta = (end-start).total_seconds()
    hours = int(delta/3600)
    minutes = int(delta/60) - 60*hours
    second = int(delta) - 3600*hours - 60*minutes

    print("Trained for {} clues. Elapsed time {}h {}m {}s".format(num_train,
                                                                  hours, minutes, seconds))
    print("(total seconds, just in case, {}".format(delta))

    print("Saving to ckpt")
    saver.save(sess, "ckpt")

    #do some tests
    x,y,l,xe,ye = c.next_with_english()
    #x,y,l = triples[0]
    #xe,ye = english[0]
    t = sess.run(top_letters, feed_dict={clues:x, answers:y, lengths:l})
    print("clues:\n"+"\n".join(' '.join(x) for x in xe))
    print("real answers:\n"+"\n".join(ye))
    print("our guesses:\n"+"\n".join(convert_letters(t)))
