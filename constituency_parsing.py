from nltk.tokenize import word_tokenize
from collections import OrderedDict
import time
from collections import Counter
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
from tensorflow.python.ops import lookup_ops

tf.set_random_seed(42)

vocab_file = '../data/vocab.txt'
input_file = '../data/shuf_tree_new1.txt'
# input_file = '../data/small_tree.txt'
lab_vocab_file = '../data/vocab.txt'
valid_file = '../data/cp_valid.txt'

flag = True


vocab_size = 30000
embedding_size = 256
batch_size = 32
num_neurons = 512

logging = tf.logging
logging.set_verbosity(logging.INFO)

def log_msg(msg):
   logging.info(f'{time.ctime()}: {msg}')


def create_dataset(inp_file, vocab_table,lab_vocab, batch_size):
    dataset = tf.data.TextLineDataset(inp_file)
    dataset = dataset.map(lambda sentence: (tf.string_split([sentence], '\t').values[0], tf.string_split([sentence], '\t').values[1]) )
    dataset = dataset.map(lambda sen, lab: (tf.string_split([sen]).values, tf.string_split([lab]).values) )
    dataset = dataset.map(lambda sen, lab: (sen, lab, tf.concat( (lab[1:], ['EOS']),axis = 0 )) )
    dataset = dataset.map(lambda sen, lab1, lab2:  (vocab_table.lookup(sen), lab_vocab.lookup(lab1), lab_vocab.lookup(lab2)  )  )
    dataset = dataset.map(lambda sen, lab1, lab2: (sen, tf.size(sen), lab1, lab2, tf.size(lab1)) )
    dataset = dataset.padded_batch(batch_size = batch_size, padded_shapes = ([None], [] ,[None], [None] , []))
    return dataset


class Embedding(tf.keras.Model):
    def __init__(self, V, d, init):
        super(Embedding, self).__init__()
#         self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
        self.W = tfe.Variable(init)
    
    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)

class StaticRNN(tf.keras.Model):
    def __init__(self, h, cell):
        super(StaticRNN, self).__init__()
        if cell == 'lstm':
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=h)
        elif cell == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=h)
        else:
            self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=h)
        
        
    def call(self, word_vectors, num_words, state, init_state):
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        if state:
            outputs, final_state = tf.nn.static_rnn(cell=self.cell, initial_state = init_state, sequence_length=num_words, inputs=word_vectors_time, dtype=tf.float32)
        else:
            outputs, final_state = tf.nn.static_rnn(cell=self.cell,  sequence_length=num_words, inputs=word_vectors_time, dtype=tf.float32)
        return outputs, final_state


class Encoder(tf.keras.Model):
    def __init__(self, V, d, h, cell):
        super(Encoder, self).__init__()
        init = tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d])
        self.word_embedding = Embedding(V, d, init)
        self.rnn = StaticRNN(h, cell)

        
    def call(self, datum, lens, state, init_state):
        word_vectors = self.word_embedding(datum)        
        logits, final_state = self.rnn(word_vectors, lens, state, init_state)
        batch_outputs = []
        for i in range(int(tf.size(lens))):
            sen_len = int(lens[i])
            batch_outputs.append(logits[sen_len-1][i])

#         return logits[-1], final_state
        return tf.convert_to_tensor(batch_outputs), final_state


class Decoder(tf.keras.Model):
    def __init__(self, V, d, num_neurons):
        super(Decoder, self).__init__()
        init = encoder.weights[0]
#         init = tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d])
        self.word_embedding = Embedding(V, d, init)
        self.rnn = StaticRNN(num_neurons, 'gru')
        self.fc = tf.keras.layers.Dense(units=V)
    
    def call(self, datum, lens, state, init_state):
        word_vectors = self.word_embedding(datum)
        rnn_outputs, state = self.rnn(word_vectors, lens, state, init_state)
        rnn_outputs_stacked = tf.stack(rnn_outputs, axis=1)
        logits = self.fc(rnn_outputs_stacked)
        
        return logits
#         rnn_outputs = self.rnn(word_vectors, lens, state, init_state)
#         print(rnn_outputs)
        
        
    
def cp_loss_fun(model1, model2, datum):
    u, s = model1(datum[0], datum[1], False, dummy_state)
    logits = model2(datum[2], datum[4], True, s)
    mask = tf.sequence_mask(datum[4], dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[3]) * mask
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(datum[4]), dtype=tf.float32)

def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)

def compute_ppl(model1, model2 , dataset):
    total_loss = 0.
    total_words = 0
    for batch_num, datum in enumerate(dataset):
        avg_loss = cp_loss_fun(model1, model2 , datum)
        total_loss = avg_loss * tf.cast(tf.reduce_sum(datum[4]), dtype=tf.float32)
        total_words += tf.cast(tf.reduce_sum(datum[4]), dtype=tf.float32)
        
#         if batch_num % 50 == 0:
#             print(f'ppl Done batch: {batch_num}')
    loss = total_loss / total_words
    return np.exp(loss)

encoder = Encoder(vocab_size, embedding_size, num_neurons, 'gru')
decoder = Decoder(vocab_size, embedding_size, num_neurons)


vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value = 0)
lab_vocab = lookup_ops.index_table_from_file(lab_vocab_file, default_value = 0)
opt = tf.train.AdamOptimizer(learning_rate=0.002)


checkpoint_dir = '../cp_encoder'
root = tfe.Checkpoint(optimizer=opt, model=encoder, optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))


root1 = tfe.Checkpoint(optimizer=opt, model=decoder, optimizer_step=tf.train.get_or_create_global_step())

if flag:
	root1.restore(tf.train.latest_checkpoint(checkpoint_dir))



dataset = create_dataset(input_file, vocab_table, lab_vocab ,batch_size)
valid_dataset = create_dataset(valid_file, vocab_table, lab_vocab ,batch_size)

dummy_state = tf.convert_to_tensor(np.zeros(num_neurons))



opt = tf.train.AdamOptimizer(learning_rate=0.002)
loss_and_grads_fun = tfe.implicit_value_and_gradients(cp_loss_fun)

checkpoint_dir = '../cp_encoder'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
root = tfe.Checkpoint(optimizer=opt, model=encoder, optimizer_step=tf.train.get_or_create_global_step())



NUM_EPOCHS = 1
STATS_STEPS = 20
EVAL_STEPS = 20

valid_ppl = compute_ppl(encoder, decoder, valid_dataset)
print(f'Start :Valid ppl: {valid_ppl}')

for epoch_num in range(NUM_EPOCHS):
    batch_loss = []
    # dataset = dataset.shuffle(buffer_size = 1000)
    for step_num, datum in enumerate(dataset, start=1):

        loss_value, gradients = loss_and_grads_fun(encoder, decoder, datum)
        batch_loss.append(loss_value)

        if step_num % STATS_STEPS == 0:
            print(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))}')
            batch_loss = []

        if step_num % EVAL_STEPS == 0:
            ppl = compute_ppl(encoder, decoder, valid_dataset)
            #Save model!
            if ppl < valid_ppl:
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl: 0.4f}')   
                save_path = root.save(checkpoint_prefix)
                save_path = root1.save(checkpoint_prefix)
#                 print(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl} old: {valid_ppl} Model saved: {save_path}')
                valid_ppl = ppl
            else:
                print(f'Epoch: {epoch_num} Step: {step_num} ppl worse: {ppl} old: {valid_ppl}')



        opt.apply_gradients(clip_gradients(gradients, 5.0), global_step=tf.train.get_or_create_global_step())

    print(f'Epoch{epoch_num} Done!')