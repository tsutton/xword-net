import tensorflow as tf
from tensorflow.python.util import nest as tf_nest

# lstm and gru reference - https://arxiv.org/pdf/1412.3555.pdf

# future:
# - Variable initialization
# - dtype option
class MyGRUCell:
    def __init__(self, batch_size, hidden_dim, input_dim, output_dim):

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # create variables
        with tf.variable_scope("GRUCell"):
            self.input_to_reset = tf.get_variable("input_to_reset", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_reset = tf.get_variable("state_to_reset", shape=(hidden_dim,hidden_dim), dtype=tf.float32)

            self.input_to_update = tf.get_variable("input_to_update", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_update = tf.get_variable("state_to_update", shape=(hidden_dim,hidden_dim), dtype=tf.float32)
        
            self.input_to_act = tf.get_variable("input_to_act", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_act = tf.get_variable("state_to_act", shape=(hidden_dim,hidden_dim), dtype=tf.float32)
            
            self.state_to_output = tf.get_variable("state_to_output", shape=(hidden_dim,output_dim), dtype=tf.float32)

    def zero_state(self):
        return tf.zeros((self.batch_size, self.hidden_dim), dtype=tf.float32)

    def zero_output(self):
        return tf.zeros((self.batch_size, self.output_dim), dtype=tf.float32)

    def __call__(self, inputs, state):
        # inputs: (batch_size, input_dim)
        # state:  (batch_size, hidden_dim)

        # r: (batch_size, hidden_dim)
        r = tf.sigmoid(tf.matmul(inputs, self.input_to_reset) + tf.matmul(state, self.state_to_reset))

        # h_tilde: (batch_size, hidden_dim)
        rh = r * state
        h_tilde = tf.tanh(tf.matmul(inputs, self.input_to_act) + tf.matmul(rh, self.state_to_act))

        # z: (batch_size, hidden_dim)
        z = tf.sigmoid(tf.matmul(inputs, self.input_to_update) + tf.matmul(state, self.state_to_update))

        # h: (batch_size, hidden_dim)
        ones = tf.constant(1.0, dtype=tf.float32, shape=z.shape)
        h = (ones - z) * state + z * h_tilde
        
        # output: (batch_size, output_dim)
        output = tf.matmul(h, self.state_to_output)
        
        return output, h


class MyLSTMCell:
    def __init__(self, batch_size, hidden_dim, input_dim, output_dim):

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim # hidden_dim = memory_dim, total state dim is the sum
        self.output_dim = output_dim

        # create variables
        with tf.variable_scope("LSTMCell"):
            # gates: input, forget, output, memory
            # output, input, forget (not memory) have a component FROM memory

            self.input_to_input = tf.get_variable("input_to_input", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_input = tf.get_variable("state_to_input", shape=(hidden_dim,hidden_dim), dtype=tf.float32)
            self.memory_to_input = tf.get_variable("memory_to_input", shape=(hidden_dim,), dtype=tf.float32)

            self.input_to_output = tf.get_variable("input_to_output", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_output = tf.get_variable("state_to_output", shape=(hidden_dim,hidden_dim), dtype=tf.float32)
            self.memory_to_output = tf.get_variable("memory_to_output", shape=(hidden_dim,), dtype=tf.float32)

            self.input_to_forget = tf.get_variable("input_to_forget", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_forget = tf.get_variable("state_to_forget", shape=(hidden_dim,hidden_dim), dtype=tf.float32)
            self.memory_to_forget = tf.get_variable("memory_to_forget", shape=(hidden_dim,), dtype=tf.float32)
            
            self.input_to_memory = tf.get_variable("input_to_memory", shape=(input_dim,hidden_dim), dtype=tf.float32)
            self.state_to_memory = tf.get_variable("state_to_memory", shape=(hidden_dim,hidden_dim), dtype=tf.float32)

            self.state_to_final = tf.get_variable("state_to_final", shape=(hidden_dim,output_dim), dtype=tf.float32)

    def zero_state(self):
        return (tf.zeros((self.batch_size, self.hidden_dim), dtype=tf.float32),
                tf.zeros((self.batch_size, self.hidden_dim), dtype=tf.float32))

    def zero_output(self):
        return tf.zeros((self.batch_size, self.output_dim), dtype=tf.float32)

    def __call__(self, inputs, state):
        # inputs: (batch_size, input_dim)
        # state: tuple ( h=(batch_size, hidden_dim), c=(batch_size,hidden_dim))

        h_old, c_old = state

        # f, i: (batch_size, hidden_dim)
        f = tf.sigmoid(tf.matmul(inputs, self.input_to_forget) +
                       tf.matmul(h_old, self.state_to_forget) +
                       (c_old * self.memory_to_forget) )

        i = tf.sigmoid(tf.matmul(inputs, self.input_to_input) +
                       tf.matmul(h_old, self.state_to_input) +
                       (c_old * self.memory_to_input) )

        # c_tilde, c: (batch_size, hidden_dim)
        c_tilde = tf.tanh(tf.matmul(inputs, self.input_to_memory) + tf.matmul(h_old, self.state_to_memory))

        c = f * c_old + i * c_tilde

        # o: (batch_size, hidden_dim)
        o = tf.sigmoid(tf.matmul(inputs, self.input_to_output) +
                       tf.matmul(h_old , self.state_to_output) +
                       (c_old * self.memory_to_output) )

        h = o * tf.tanh(c)
        
        # output: (batch_size, output_dim)
        output = tf.matmul(h, self.state_to_final)
        
        return output, (h, c)

class SkipZerosWrapper:

    def __init__(self, cell):
        self.cell = cell

    def zero_state(self):
        return self.cell.zero_state()

    def zero_output(self):
        return self.cell.zero_output()

    def __call__(self, inputs, state):

        # we need to do some gymnastics because cond expects its pieces to return
        # lists of tensors, and we have, well, something else: (output, state), and state may be nested

        nest_template = (inputs, state)

        def true_fn():
            return tf_nest.flatten(self.cell(inputs, state))
        def false_fn():
            return tf_nest.flatten( (inputs, state) )

        l = tf.cond(tf.reduce_any(tf.cast(inputs, tf.bool)),
                    true_fn,
                    false_fn)
        
        return tf_nest.pack_sequence_as(nest_template, l)
