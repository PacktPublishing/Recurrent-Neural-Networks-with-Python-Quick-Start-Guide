import time
import tensorflow as tf
import tensorlayer as tl
from sklearn.utils import shuffle
from tensorlayer.layers import EmbeddingInputlayer, Seq2Seq, DenseLayer, retrieve_seq_length_op2

from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

# Hyperparameters
batch_size = 32
embedding_dimension = 1024
learning_rate = 0.0001
number_epochs = 1000

xseq_len = len(trainX)
yseq_len = len(trainY)
assert xseq_len == yseq_len

n_step = int(xseq_len/batch_size)

w2idx = metadata['w2idx']
idx2w = metadata['idx2w']

xvocab_size = len(metadata['idx2w']) # 8002 (0~8001)
start_id = xvocab_size  # 8002
end_id = xvocab_size+1  # 8003

w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
idx2w = idx2w + ['start_id', 'end_id']

xvocab_size = yvocab_size = xvocab_size + 2

# Model
def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = embedding_dimension,
                name = 'seq_embedding')
            vs.reuse_variables()
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = embedding_dimension,
                name = 'seq_embedding')
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.contrib.rnn.BasicLSTMCell,
                n_hidden = embedding_dimension,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
    return net_out, net_rnn

# Initialize model for training
encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")
net_out, _ = model(encode_seqs, decode_seqs, is_train=True, reuse=False)

loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask, name='cost')

#net_out.print_params(False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Understand tf.ConfigProto - https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())

# Train
def train():
    print("Start training")
    for epoch in range(number_epochs):
        epoch_time = time.time()
        trainX_shuffled, trainY_shuffled = shuffle(trainX, trainY, random_state=0)
        total_err, n_iter = 0, 0
        for X, Y in tl.iterate.minibatches(inputs=trainX_shuffled, targets=trainY_shuffled, batch_size=batch_size, shuffle=False):

            X = tl.prepro.pad_sequences(X)

            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)

            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            _, err = sess.run([optimizer, loss],
                            {encode_seqs: X,
                            decode_seqs: _decode_seqs,
                            target_seqs: _target_seqs,
                            target_mask: _target_mask})

            if n_iter % 200 == 0:
                print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, number_epochs, n_iter, n_step, err, time.time() - epoch_time))

            total_err += err; n_iter += 1

# Initialize model for prediction
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
net, net_rnn = model(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
y = tf.nn.softmax(net.outputs)

# Predict
def predict():
    seeds = ["happy birthday have a nice day",
            "donald trump won last nights presidential debate according to snap online polls"]
    for seed in seeds:
        seed_id = [w2idx[w] for w in seed.split(" ")]
        for _ in range(5):  # 1 Query --> 5 Reply
            # 1. encode, get state
            state = sess.run(net_rnn.final_state_encode,
                            {encode_seqs2: [seed_id]})
            # 2. decode, feed start_id, get first word
            o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                            decode_seqs2: [[start_id]]})
            w_id = tl.nlp.sample_top(o[0], top_k=3)
            w = idx2w[w_id]
            # 3. decode, feed state iteratively
            sentence = [w]
            for _ in range(30): # max sentence length
                o, state = sess.run([y, net_rnn.final_state_decode],
                                {net_rnn.initial_state_decode: state,
                                decode_seqs2: [[w_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=2)
                w = idx2w[w_id]
                if w_id == end_id:
                    break
                sentence = sentence + [w]
            print(" >", ' '.join(sentence))


train()
predict()

