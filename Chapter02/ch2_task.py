import tensorflow as tf
import random

num_examples = 100000
num_classes = 50

def input_values():
    multiple_values = [map(int, '{0:050b}'.format(i)) for i in range(2**20)]
    random.shuffle(multiple_values)
    final_values = []

    for value in multiple_values[:num_examples]:
        temp = []
        for number in value:
            temp.append([number])
        final_values.append(temp)

    return final_values

def output_values(inputs):
    final_values = []
    for value in inputs:
        output_values = [0 for _ in range(num_classes)]
        count = 0
        for i in value:
            count += i[0]
        if count < num_classes:
            output_values[count] = 1
        final_values.append(output_values)
    return final_values

def generate_data():
    inputs = input_values()
    return inputs, output_values(inputs)

X = tf.placeholder(tf.float32, shape=[None, num_classes, 1])

Y = tf.placeholder(tf.float32, shape=[None, num_classes])

num_hidden_units = 24

weights = tf.Variable(tf.truncated_normal([num_hidden_units, num_classes]))

biases = tf.Variable(tf.truncated_normal([num_classes]))

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=num_hidden_units)

outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs=X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])

last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

prediction = tf.matmul(last_output, weights) + biases

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=prediction)

total_loss = tf.reduce_mean(loss)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)


batch_size = 1000

number_of_batches = int(num_examples/batch_size)

epoch = 1000

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    X_train, y_train = generate_data()

    for i in range(epoch):
        iter = 0

        for epoch in range(number_of_batches):

            training_x = X_train[iter:iter+batch_size]

            training_y = y_train[iter:iter+batch_size]

            iter += batch_size

            _, current_total_loss = sess.run([optimizer, total_loss], feed_dict={X: training_x, Y: training_y})

            print("Iteration", iter, "loss", current_total_loss)

            print("__________________")

    # Generate
    test_example = [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],

                    [1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],

                    [1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]

    prediction_result = sess.run(prediction, {X: test_example})

    largest_number_index = prediction_result[0].argsort()[-1:][::-1]

    print("Predicted sum: ", largest_number_index, "Actual sum:", 30)
    print("The predicted sequence parity is ", largest_number_index % 2, " and it should be: ", 0)
