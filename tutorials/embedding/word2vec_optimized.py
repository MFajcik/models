# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.

Additional edits made by Martin Fajcik
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import pickle
import datetime
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
                       "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 300, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 1,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 1,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 1,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

flags.DEFINE_boolean("save2vec",False,"Save model into .vec file after training.")
flags.DEFINE_boolean("save_dict_pkl",False,"Pickle the word to index dictionary.")
flags.DEFINE_boolean("make_topgrads_pkl",False,"Save matrix containt information about topgradients into pkl.")

flags.DEFINE_boolean("validate",False,"Run validation on word analogy task during training [EN only].")
flags.DEFINE_boolean("save_checkpoints",True,"Save network architecture and parameters after each epoch.")
flags.DEFINE_boolean("playmode", False, "Enter mode in which checkpoint from save_path is loaded and ipython console is enabled.")

FLAGS = flags.FLAGS


class Options(object):
    """Options used by our word2vec model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # Training options.

        # The training text file.
        self.train_data = FLAGS.train_data

        # Number of negative samples per example.
        self.num_samples = FLAGS.num_neg_samples

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = FLAGS.min_count

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Eval options.

        # The text file for eval.
        self.eval_data = FLAGS.eval_data


ladder_size = 1024
class Word2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_get_embedding_graph()
        self.build_get_pos_graph()
        self.build_find_nearest_graph()
        # eval graph also runs the variable initializer!
        self.build_eval_graph()
        print ("Saving vocabulary...")
        self.save_vocab()
        print ("Done")

        #self.examples = np.zeros(shape=(self._options.vocab_size, ladder_size),dtype=np.int32)
        self.gradient_positions = np.zeros(shape=(self._options.vocab_size, ladder_size), dtype=np.uint64)
        self.gradients = np.zeros(shape=(self._options.vocab_size, ladder_size),dtype=np.float)

    # The vec file is a text file that contains the word vectors, one per line for each word in the vocabulary.
    # The first line is a header containing the number of words and the dimensionality of the vectors.
    # Subsequent lines are the word vectors for all words in the vocabulary, sorted by decreasing frequency.
    # Example:
    # 218316 100
    # the -0.10363 -0.063669 0.032436 -0.040798...
    # of -0.0083724 0.0059414 -0.046618 -0.072735...
    # one 0.32731 0.044409 -0.46484 0.14716...
    def save_to_vec(self, vec_path):
        embeddings = self._session.run(self._w_in)
        # Using linux file endings
        with open(vec_path, 'w') as f:
            print("Saving .vec file to {}".format(vec_path))
            f.write("{} {}\n".format(self._options.vocab_size, self._options.emb_dim))
            for (word, embedding) in zip(self._options.vocab_words, embeddings):
                f.write("{} {}\n".format(word, ' '.join(map(str, embedding))))

    def read_analogies(self):
        """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", self._options.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)

    def build_graph(self):
        #with tf.name_scope('graph') as scope:

            """Build the model graph."""
            opts = self._options

            # The training data. A text file.
            #print("Preprocessing  corpus...")
            (words, counts, words_per_epoch, current_epoch, total_words_processed,
             examples, labels, example_positions, label_positions) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                            batch_size=opts.batch_size,
                                                            window_size=opts.window_size,
                                                            min_count=opts.min_count,
                                                            subsample=opts.subsample,
                                                            gradient_ranking=True)
            (opts.vocab_words, opts.vocab_counts,
             opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
            #print("Preprocessing finished!")
            opts.vocab_words = list(map(lambda s: s.decode(),opts.vocab_words))
            #print("Words saved!")
            opts.vocab_size = len(opts.vocab_words)
            print("Data file: ", opts.train_data)
            print("Vocab size: ", opts.vocab_size - 1, " + UNK")
            print("Words per epoch: ", opts.words_per_epoch)

            self._id2word = opts.vocab_words
            for i, w in enumerate(self._id2word):
                self._word2id[w] = i

            # Declare all variables we need.
            # Input words embedding: [vocab_size, emb_dim]
            w_in = tf.Variable(
                tf.random_uniform(
                    [opts.vocab_size,
                     opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
                name="w_in")

            pos_by_gradient_ladder = tf.Variable(
                tf.zeros(shape=(opts.vocab_size, ladder_size),name="pos_by_gradient_ladder",dtype=tf.int32),
            )
            gradient_ladder = tf.Variable(
                tf.zeros(shape=(opts.vocab_size, ladder_size), name="gradient_ladder")
            )
            #
            # example_ladder = tf.Variable(
            #     tf.zeros(shape=(ladder_size), name="example_ladder",dtype=tf.int32)
            # )

            # Global step: scalar, i.e., shape [].
            w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

            # Global step: []
            global_step = tf.Variable(0, name="global_step")

            # Linear learning rate decay.
            words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
            lr = opts.learning_rate * tf.maximum(
                0.0001,
                1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

            # Training nodes.
            inc = global_step.assign_add(1)
            with tf.control_dependencies([inc]):
                train = word2vec.neg_train_word2vec(w_in,
                                                    w_out,
                                                    examples,
                                                    labels,
                                                    example_positions,
                                                    label_positions,
                                                    lr,
                                                    pos_by_gradient_ladder,
                                                    gradient_ladder,
                                                    vocab_count=opts.vocab_counts.tolist(),
                                                    num_negative_samples=opts.num_samples)

            self._w_in = w_in
            self._examples = examples
            self._labels = labels
            self._lr = lr
            self._train = train
            self.global_step = global_step
            self._epoch = current_epoch
            self._words = total_words_processed
            self._pos_by_gradient_ladder = pos_by_gradient_ladder
            self._gradient_ladder = gradient_ladder

    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i])
                f.write("%s %d\n" % (vocab_word,
                                     opts.vocab_counts[i]))

    def build_get_embedding_graph(self):
        word_id = tf.placeholder(dtype=tf.int32)
        emb = tf.gather(self._w_in, word_id)

        self.embedding = emb
        self._word_id = word_id

    def build_get_pos_graph(self):
        # word_id = tf.placeholder(dtype=tf.int32)
        # pos_graph =  tf.gather(self._pos_by_gradient_ladder, word_id)
        # self.pos_graph = pos_graph
        # self._word_id = word_id
        pass
    def get_pos(self,w):
        #return self._session.run([self.pos_graph], {self._word_id: self._word2id.get(w, 0)})[0]
        return self.gradient_positions[self._word2id[w]]

    def build_find_nearest_graph(self, nearest_word_count=10):
        opts=self._options
        word_id = tf.placeholder(dtype=tf.int32)
        normalized_embedding_matrix = tf.nn.l2_normalize(self._w_in, 1)
        embedding = tf.gather(normalized_embedding_matrix, word_id)
        distance_matrix = tf.matmul(tf.expand_dims(embedding,0), normalized_embedding_matrix, transpose_b=True)
        #distance_matrix= tf.einsum('n,nm->m', embedding, tf.transpose(normalized_embedding_matrix))
        _, closest_idx = tf.nn.top_k(distance_matrix, min(nearest_word_count+1, opts.vocab_size))
        self.closest_idx=closest_idx
        self.target_id = word_id


    def build_eval_graph(self):
        """Build the evaluation graph."""
        # Eval graph
        opts = self._options

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        nemb = tf.nn.l2_normalize(self._w_in, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, opts.vocab_size))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()


    def find_insert_pos(self,example,gradient):
        if gradient<self.gradients[example,-1]:
            return -1
        high = ladder_size-1
        low = 0
        while low <= high:
            mid = (high + low) // 2
            mid_val = self.gradients[example, mid]
            if(mid_val > gradient): low = mid + 1
            else: high = mid - 1
        return low


    def process_result(self,examples,positions,gradients):
        #shape=(self._options.vocab_size, ladder_size)
        for i in range(len(examples)):
            example,position,gradient = examples[i],positions[i],gradients[i]
            #find pos to insert via binary search
            p = self.find_insert_pos(example,gradient)
            if p>=0:
                #self.examples[example,shift_from+1:] = self.examples[example,shift_from:-1]
                self.gradient_positions[example, p + 1:] = self.gradient_positions[example, p:-1]
                self.gradients[example,p+1:] = self.gradients[example,p:-1]
                self.gradient_positions[example, p] = position
                self.gradients[example,p] = gradient

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
           (examples,positions,gradients),epoch = self._session.run([self._train, self._epoch])
           self.process_result(examples,positions,gradients)
           if epoch != initial_epoch:
               break

    def train(self):
        """Train the model."""
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time = initial_words, time.time()
        while True:
            time.sleep(5)  # Reports our progress once a while.
            (epoch, step, words, lr) = self._session.run(
                [self._epoch, self.global_step, self._words, self._lr])
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (
                    now - last_time)
            # print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
            #                                                               lr, rate),
            #       end="")
            # sys.stdout.flush()
            progress = words / (opts.words_per_epoch*opts.epochs_to_train) * 100
            print("[%3.2f%%]: Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (progress,epoch, step,
                                                                          lr, rate))
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()
    def _find_nearest(self, w):
        idx, = self._session.run(self.closest_idx, {
            self.target_id:self._word2id.get(w,0)
        })
        result = list(map(lambda x: self._id2word[x],idx[1:]))
        return result

    def _predict(self, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx

    def get_embedding(self, word):
        return self._session.run([self.embedding], {self._word_id: self._word2id.get(word, 0)})

    def eval(self):
        """Evaluate analogy questions and reports accuracy."""

        # How many questions we get right at precision@1.
        correct = 0
        try:
            total = self._analogy_questions.shape[0]
        except AttributeError   as e:
            raise AttributeError("Need to read analogy questions.")

        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                        # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break
        print()
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,

                                                  correct * 100.0 / total))

    def analogy(self, w0, w1, w2):
        """Predict word w3 as in w0:w1 vs w2:w3."""
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                print(c)
                break
        print("unknown")

    def nearby(self, words, num=20):
        """Prints out nearby words given a list of words."""
        ids = np.array([self._word2id.get(x, 0) for x in words])
        vals, idx = self._session.run(
            [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in xrange(len(words)):
            print("\n%s\n=====================================" % (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self._id2word[neighbor], distance))


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


#FLAGS.train_data = "/mnt/minerva1/nlp/projects/semantic_relatedness10/data/my_preprocessed/cs.txt_2017-10-25_12:19"
#FLAGS.train_data = "/home/ifajcik/deep_learning/word2vec/corpus_data/ebooks_corpus_CZ/few_sentences.txt"
FLAGS.train_data = "/home/ifajcik/deep_learning/word2vec/corpus_data/ebooks_corpus_CZ/e_knihy_preprocessed.txt"
#FLAGS.train_data = "/home/ifajcik/deep_learning/word2vec/corpus_data/cwc_corpus2011/cwc50megs"
FLAGS.eval_data = "noevaltest"
#FLAGS.save_path = "/home/ifajcik/word2vec/trainedmodels/tf_w2vopt_zoznam"
FLAGS.save_path = "/home/ifajcik/deep_learning/word2vec/trainedmodels/tf_w2vopt_ebooks_gradient_ladder"

def max (x,y):
    return x if x>y else y

def find_word_contexts(model, word, window_size =FLAGS.window_size * 2, corpus =FLAGS.train_data, average_word_bytelen = 45):
    positions = model.get_pos(word)
    size = window_size * average_word_bytelen #10 for average word
    contexts = []
    with open(corpus,mode="rb") as data:
        for pos in positions:
            data.seek(max(int(pos-size),0))
            chunk1 = data.read(size).decode("utf-8", errors="ignore").split()
            chunk2 = data.read(size).decode("utf-8", errors="ignore").split()
            contexts.append(chunk1[-(window_size+1):]+chunk2[:window_size])
    return contexts,positions

def disable_GPU():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main(_):
    disable_GPU()
    """Train a word2vec model."""
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()

    if FLAGS.playmode:
        play_with_model(opts)

    model_n = os.path.basename(FLAGS.train_data)
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Word2Vec(opts, session)
            if FLAGS.validate:
                model.read_analogies()  # Read analogy questions
            print("Starting training: {}".format(datetime.date.today()))
            print("Epochs to train: {}".format(opts.epochs_to_train))
            for i in range(opts.epochs_to_train):
                model.train()  # Process one epoch
                if FLAGS.validate:
                    model.eval()
                if FLAGS.save_checkpoints:
                    model.saver.save(session, os.path.join(opts.save_path, "model_{}_e{}.ckpt".format(model_n,i)),
                           global_step=model.global_step)

        print("Training finished: {}".format(datetime.date.today()))

        if FLAGS.save_dict_pkl:
            with open ("w2i_{}.pkl".format(model_n), "wb") as f:
                pickle.dump(model._word2id, f, pickle.HIGHEST_PROTOCOL)

        if FLAGS.make_topgrads_pkl:
            with open('toppositions_{}.pkl'.format(model_n), 'wb') as output:
                pickle.dump(model.gradient_positions, output, pickle.HIGHEST_PROTOCOL)

            with open('topgrads_{}.pkl'.format(model_n), 'wb') as output:
                pickle.dump(model.gradients, output, pickle.HIGHEST_PROTOCOL)

        if FLAGS.save2vec:
            save_model_to_vec(opts)
        if FLAGS.interactive:
            # E.g.,
            # [0]: model.analogy(b'france', b'paris', b'russia')
            # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
            _start_shell(locals())


def save_model_to_vec(opts, restore_from_save_path=False):
    with tf.Session() as session:
        model = Word2Vec(opts, session)
        if restore_from_save_path:
            print('Restoring model...')
            model_n = os.path.basename(FLAGS.train_data)
            model.saver.restore(session, tf.train.latest_checkpoint(FLAGS.save_path))
        model.save_to_vec(os.path.join(FLAGS.save_path, "{}_model.vec".format(model_n)))
    return True

def play_with_model(opts):
    with tf.Session() as session:
        model = Word2Vec(opts, session)
        print('Restoring model...')
        model.saver.restore(session, tf.train.latest_checkpoint(opts.save_path))

        #nearest_words = model._find_nearest('oko')
        #print(' '.join(nearest_words))
        _start_shell(locals())
    return True

if __name__ == "__main__":
    tf.app.run()