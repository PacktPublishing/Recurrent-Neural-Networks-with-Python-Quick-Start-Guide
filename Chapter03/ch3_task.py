import numpy as np
import tensorflow as tf
import sys
import collections

def get_words(file_name):
    with open(file_name) as file:
        all_lines = file.readlines()
    lines_without_spaces = [x.strip() for x in all_lines]
    words = []
    for line in lines_without_spaces:
        words.extend(line.split())
    words = np.array(words)
    return words

def build_dictionary(words):
    most_common_words = collections.Counter(words).most_common()
    word2id = dict((word, id) for (id, (word, _)) in enumerate(most_common_words))
    id2word = dict((id, word) for (id, (word, _)) in enumerate(most_common_words))
    return most_common_words, word2id, id2word

words = get_words("./the_hunger_games.txt")

# Check if text is added to `the_hunger_games.txt` file

if len(words) < 2000:
    print("> Please enter text with more than 2000 words inside `the_hunger_games.txt` file. Then run the program again.")
    sys.exit()

most_common_words, word2id, id2word = build_dictionary(words)
most_common_words_length = len(most_common_words)

section_length = 20

def input_output_values():
    input_values = []
    output_values = []

    for i in range(len(words) - section_length):
        input_values.append(words[i: i + section_length])
        output_values.append(words[i + section_length])

    one_hot_inputs = np.zeros((len(input_values), section_length, most_common_words_length))
    one_hot_outputs = np.zeros((len(output_values), most_common_words_length))

    for s_index, section in enumerate(input_values):
        for w_index, word in enumerate(section):
            one_hot_inputs[s_index, w_index, word2id[word]] = 1.0
        one_hot_outputs[s_index, word2id[output_values[s_index]]] = 1.0

    return one_hot_inputs, one_hot_outputs

training_X, training_y = input_output_values()

learning_rate = 0.001
batch_size = 512
number_of_iterations = 100000
number_hidden_units = 1024

X = tf.placeholder(tf.float32, shape=[batch_size, section_length, most_common_words_length])
y = tf.placeholder(tf.float32, shape=[batch_size, most_common_words_length])

weights = tf.Variable(tf.truncated_normal([number_hidden_units, most_common_words_length]))
biases = tf.Variable(tf.truncated_normal([most_common_words_length]))

gru_cell = tf.contrib.rnn.GRUCell(num_units=number_hidden_units)

outputs, state = tf.nn.dynamic_rnn(gru_cell, inputs=X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])

last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
prediction = tf.matmul(last_output, weights) + biases

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
total_loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iter_offset = 0
    saver = tf.train.Saver()
    for iter in range(number_of_iterations):
        length_X = len(training_X)

        if length_X != 0:
            iter_offset = iter_offset % length_X

        if iter_offset <= length_X - batch_size:
            training_X_batch = training_X[iter_offset: iter_offset + batch_size]
            training_y_batch = training_y[iter_offset: iter_offset + batch_size]
            iter_offset += batch_size
        else:
            add_from_the_beginning = batch_size - (length_X - iter_offset)
            print("Training_X:", sess.run(tf.shape(training_X)), "X:", sess.run(tf.shape(X[0: 3])))
            training_X_batch = np.concatenate((training_X[iter_offset: length_X], X[0: add_from_the_beginning]))
            training_y_batch = np.concatenate((training_y[iter_offset: length_X], y[0: add_from_the_beginning]))
            iter_offset = add_from_the_beginning

        _, training_loss = sess.run([optimizer, total_loss], feed_dict={X: training_X_batch, y: training_y_batch})
        if iter % 10 == 0:
            print("Loss:", training_loss)
            saver.save(sess, 'ckpt/model', global_step=iter)

def prediction_to_one_hot(prediction):
    zero_array = np.zeros(np.shape(prediction))
    zero_array[np.argmax(prediction)] = 1
    return zero_array

# Generate
starting_sentence = 'I plan to make the world a better place because I love seeing how people grow and do in their lives '
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = tf.train.latest_checkpoint('ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, model)

    generated_text = starting_sentence
    words_in_starting_sentence = starting_sentence.split()
    test_X = np.zeros((1, section_length, most_common_words_length))

    for index, word in enumerate(words_in_starting_sentence[:-1]):
        if index < section_length:
            test_X[0, index, word2id[word]] = 1

    _ = sess.run(prediction, feed_dict={X: test_X})

    test_last_X = np.zeros((1, 1, most_common_words_length))
    test_last_X[0, 0, word2id[words_in_starting_sentence[-1]]] = 1
    test_next_X = np.reshape(np.concatenate((test_X[0, 1:], test_last_X[0])), (1, section_length, most_common_words_length))

    # Generate the new 200 words
    for i in range(1000):
        test_prediction = prediction.eval({X: test_next_X})[0]
        next_word_one_hot = prediction_to_one_hot(test_prediction)
        next_word = id2word[np.argmax(next_word_one_hot)]
        generated_text += next_word + " "
        test_next_X = np.reshape(np.concatenate((test_next_X[0, 1:], np.reshape(next_word_one_hot, (1, most_common_words_length)))), (1, section_length, most_common_words_length))

    print("Generated text: ", generated_text)
