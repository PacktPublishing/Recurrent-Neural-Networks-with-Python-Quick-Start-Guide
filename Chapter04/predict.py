import tensorflow as tf
import numpy as np
import neural_machine_translation as nmt

# Placeholders
encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(nmt.input_sequence_length)]
decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(nmt.output_sequence_length)]

with tf.variable_scope("model_params", reuse=True):
    w_t = tf.get_variable('proj_w', [nmt.english_vocab_size, nmt.size], tf.float32)
    b = tf.get_variable('proj_b', [nmt.english_vocab_size], tf.float32)
    w = tf.transpose(w_t)
    output_projection = (w, b)

    outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                encoder_inputs,
                                                decoder_inputs,
                                                tf.contrib.rnn.BasicLSTMCell(nmt.size),
                                                num_encoder_symbols = nmt.spanish_vocab_size,
                                                num_decoder_symbols = nmt.english_vocab_size,
                                                embedding_size = nmt.embedding_size,
                                                feed_previous = True,
                                                output_projection = output_projection,
                                                dtype = tf.float32)

# ops for projecting outputs
outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(nmt.output_sequence_length)]

# let's translate these sentences
spanish_sentences = ["Como te llamas", "Mi nombre es", "Estoy leyendo un libro", "Que tal",\
                "Estoy bien", "Hablas espanol", "Que hora es", "Hola", "Adios", "Si", "No"]
spanish_sentences_encoded = [[nmt.spanish_word2idx.get(word, 0) for word in spanish_sentence.split()] for spanish_sentence in spanish_sentences]

# padding to fit encoder input
for i in range(len(spanish_sentences_encoded)):
    spanish_sentences_encoded[i] += (nmt.input_sequence_length - len(spanish_sentences_encoded[i])) * [nmt.spanish_word2idx['<pad>']]

def decode_output(output_sequence):
    words = []
    for i in range(nmt.output_sequence_length):
        smax = nmt.softmax(output_sequence[i])
        maxId = np.argmax(smax)
        words.append(nmt.english_idx2word[maxId])
    return words

# restore all variables - use the last checkpoint saved
saver = tf.train.Saver()
path = tf.train.latest_checkpoint('ckpt')

with tf.Session() as sess:
    # restore
    saver.restore(sess, path)

    # feed data into placeholders
    feed = {}
    for i in range(nmt.input_sequence_length):
        feed[encoder_inputs[i].name] = np.array([spanish_sentences_encoded[j][i] for j in range(len(spanish_sentences_encoded))], dtype = np.int32)

    feed[decoder_inputs[0].name] = np.array([nmt.english_word2idx['<go>']] * len(spanish_sentences_encoded), dtype = np.int32)
    # translate
    output_sequences = sess.run(outputs_proj, feed_dict = feed)

    # decode seq.
    for i in range(len(spanish_sentences_encoded)):
        ouput_seq = [output_sequences[j][i] for j in range(nmt.output_sequence_length)]
        #decode output sequence
        words = decode_output(ouput_seq)

        for j in range(len(words)):
            if words[j] not in ['<eos>', '<pad>', '<go>']:
                print(words[j], end=" ")

        print('\n--------------------------------')

