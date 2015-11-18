import tensorflow as tf
import numpy as np
import collections
import math

import mattmahoney_data

class bow2vec:

    def __init__(self, embedding_size, vocabulary_size, num_sampled, batch_size, valid_size=16, valid_window=100):
        # random.sample( np.arange(a), size )

        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.random.choice(np.arange(valid_window), valid_size)

        self.data_index = 0

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32,  shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Construct the variables.
            # embeddings: VxE weight matrix, uniform values from -1 to 1.
            embeddings = tf.Variable( tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0) )

            # nce weights: VxE weight matrix, truncated normal - re-sample values that are further than 2*stddev from mean. Default mean = 0.
            # nce biases - V-dimensional vector, initialized to zeros.
            ### tohle je divny protoze vahy jdou z vrstvy o emdedding_size neuronech do vrstvy o vocabulary_size neuronech takze je to opacne nez u embeddings
            nce_weights = tf.Variable( tf.truncated_normal(shape=[vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)) )
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
            self.loss = tf.reduce_mean( tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=self.train_labels, num_sampled=num_sampled, num_classes=vocabulary_size) )

            # Construct the SGD optimizer using a learning rate of 1.0.
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            # 1. square the whole matrix element-wise
            # 2. sum the matrix along the index 1 (the second index), result will have... index 0 is the row index -> result will be one row. index 1 is the column index -> result will be one column - sums of all rows
            # step 2 gives us one column with sums of the squared rows - sum of the squared embeddings
            # 3. take square root of the column vector. That gives us the norms of all the embeddings in one vocabulary_size-sized column vector.        
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

            # normalize embeddings: so that the embedding length is no more than one
            normalized_embeddings = embeddings / norm

            # get normalized embeddings of the words from the validation dataset (in a column vector)
            valid_embeddings = tf.nn.embedding_lookup( normalized_embeddings, valid_dataset )

            # scalar product of normalized and valid embeddings 
            # cosine similarity is just a scalar product divided by the product of the norms of the vectors - in this case, 1.
            # normalized_embeddings is a matrix of VxE
            # valid_embeddings is a matrix of |valid|xE
            # matrix product of this is a matrix of |valid| x V => similarities vectors of words from validation dataset with all words from the vocabulary
            self.similarity = tf.matmul( valid_embeddings, normalized_embeddings, transpose_b=True )



    def generate_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0     # batch is composed of num_skips-sized parts 
        assert num_skips <= 2 * skip_window    # cannot use more skip words than those from the window

        batch  = np.empty(shape=(batch_size), dtype=np.int32)    # (row) vector
        labels = np.empty(shape=(batch_size, 1), dtype=np.int32) # column vector
#        batch  = np.ndarray(shape=(batch_size), dtype=np.int32)
#        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        window = collections.deque(maxlen=span)

        def shift_data():
            next_word = self.data[self.data_index];
            self.data_index = (self.data_index + 1) % self.data_length
            return next_word

        # fill window is with words from the first window of the batch (incl. the target word)
        for _ in range(span):
            window.append(shift_data())


        for i in range(batch_size // num_skips):
            # get permutation of span to get the order in which labels are generated for the target word
            # must skip the target word

            words_in_window = np.delete(np.arange(span), skip_window)
            permutation = np.random.permutation(words_in_window)
            
            center_word = window[skip_window]

            for j in range(num_skips):

                word_in_window = window[permutation[j]]
                
                batch[i * num_skips + j] = center_word   # predict from the word in the center of the window
                labels[i * num_skips + j, 0] = word_in_window    # predict a word from the window

            # add next word to the current window (removing the last which is property of the deque)
            window.append(shift_data())

        return batch, labels




    def train(self, data, batch_size, num_skips, skip_window, reverse_dictionary):
        
        self.data = data
        self.data_length = len(data)

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            print("Initialized")

            num_steps = 100001
            average_loss = 0

            for step in xrange(num_steps):
                batch_inputs, batch_labels = self.generate_batch(batch_size, num_skips, skip_window)
                feed_dict = {self.train_inputs : batch_inputs, self.train_labels : batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = reverse_dictionary[self.valid_examples[i]]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)

            final_embeddings = normalized_embeddings.eval()







if __name__ == '__main__':
    embedding_size = 100
    vocabulary_size = 50000
    num_sampled = 64 # number of negative samples
    batch_size = 128

    model = bow2vec(embedding_size, vocabulary_size, num_sampled, batch_size)

    dataset = mattmahoney_data.dataset(vocabulary_size)
    data, count, dictionary, reverse_dictionary = dataset.load()

    num_skips = 2
    skip_window = 1

    model.train(data, batch_size, num_skips, skip_window, reverse_dictionary)

