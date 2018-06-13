from SegNetsTF import UNetDist
from SegNetsTF.Net_Utils import EarlyStopper
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from SegNetsTF.ObjectOriented import ConvolutionalNeuralNetwork
import os
from DataReadDecode import read_and_decode
from tqdm import tqdm, trange
from time import sleep
from datetime import datetime

class DNNDist(UNetDist):
    def __init__(
        self,
        TF_RECORDS_train,
        TF_RECORDS_test,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_CHANNELS=1,
        STEPS=2000,
        LRSTEP=200,
        DECAY_EMA=0.9999,
        N_PRINT = 100,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True,
        WEIGHT_DECAY=0.00005,
        LOSS_FUNC=tf.nn.l2_loss,
        N_FEATURES=16,
        N_EPOCH=1,
        N_THREADS=1,
        MEAN_FILE=None,
        DROPOUT=0.5,
        EARLY_STOPPING=10):

        self.N_EPOCH = N_EPOCH
        self.N_THREADS = N_THREADS
        self.DROPOUT = DROPOUT
        self.MEAN_FILE = MEAN_FILE
        if MEAN_FILE is not None:
            MEAN_ARRAY = tf.constant(np.load(MEAN_FILE), dtype=tf.float32) # (3)
            self.MEAN_ARRAY = tf.reshape(MEAN_ARRAY, [1, 1, NUM_CHANNELS])
            self.SUB_MEAN = True
        else:
            self.SUB_MEAN = False
#        self.TF_RECORDS = TF_RECORDS
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.init_queue(TF_RECORDS_train, TF_RECORDS_test)
        ConvolutionalNeuralNetwork.__init__(self, LEARNING_RATE, K, 
            BATCH_SIZE, IMAGE_SIZE, 1, NUM_CHANNELS, 
            STEPS, LRSTEP, DECAY_EMA, N_PRINT, LOG, 
            SEED, DEBUG, WEIGHT_DECAY, LOSS_FUNC, N_FEATURES,
            EARLY_STOPPING)

    def init_queue(self, train, test):
        """
        New queues for coordinator
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('Training_queue'):
                self.data_init, self.train_iterator = read_and_decode(train, 
                                                              self.IMAGE_SIZE[0], 
                                                              self.IMAGE_SIZE[1],
                                                              self.BATCH_SIZE,
                                                              self.N_THREADS)
            with tf.name_scope('Test_queue'):

                self.data_init_test, self.test_iterator = read_and_decode(test, 
                                                              self.IMAGE_SIZE[0], 
                                                              self.IMAGE_SIZE[1],
                                                              1,
                                                              self.N_THREADS,
                                                              TRAIN=False)
        print("Queue initialized")

    def input_node_f(self):
        """
        The input node can now come from the record or can be inputed
        via a feed dict (for testing for example)
        """
        def f_true():
            train_images, train_labels = self.train_iterator.get_next()
            return train_images, train_labels

        def f_false():
            test_images, test_labels = self.test_iterator.get_next()
            return test_images, test_labels

        with tf.name_scope('Switch'):
            self.image, self.annotation = tf.cond(self.is_training, f_true, f_false)

        if self.SUB_MEAN:
            self.images_queue = self.image - self.MEAN_ARRAY
        else:
            self.images_queue = self.image
        self.image_PH = tf.placeholder_with_default(self.images_queue, shape=[None,
                                                                              None, 
                                                                              None,
                                                                              3])
        return self.image_PH

    def train(self, test_steps):
        track = "F1"
        output = os.path.join(self.LOG, "data_collector.csv")
        look_behind = self.early_stopping_max
        early_stop = EarlyStopper(track, output, maximum=look_behind)

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        self.Saver()
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer(), self.data_init, self.data_init_test)
        self.sess.run(init_op)
        self.regularize_model()

        self.Saver()
        
        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)
        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        print "self.global step", int(self.global_step.eval())
        begin = int(self.global_step.eval())
        print "begin", begin
        for step in trange(begin, steps + begin, desc='Training'):  
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

            if step % self.N_PRINT == 0 and step != 0:
                pred = np.zeros(shape=(test_steps, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]), dtype='float')
                lab  = np.zeros(shape=(test_steps, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]), dtype='float')
                loss = np.zeros(test_steps, dtype='float')
                for sub_step in trange(test_steps, desc='Testing', leave=True):
                    l, tpredictions, batch_labels = self.sess.run([self.loss, self.train_prediction, 
                                                                   self.train_labels_node], feed_dict={self.is_training:False})
                    pred[sub_step] = tpredictions[0,:,:]
                    lab[sub_step]  = batch_labels[0,:,:,0]
                    loss[sub_step] = l
                i = datetime.now()
                tqdm.write(i.strftime('%Y/%m/%d %H:%M:%S: \n '))
                self.summary_writer.add_summary(s, step)                
                tqdm.write('  Step %d of %d' % (step, steps))
                tqdm.write('  Learning rate: {} \n'.format(lr))
                tqdm.write('  Mini-batch loss: {} \n '.format(l))
                tqdm.write('  Max value: {} \n '.format(np.max(pred)))
                self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)
                wgt_path = self.LOG + '/' + "model.ckpt-{}".format(step)
                values_test = [np.mean(loss), self.Validation(pred, lab), wgt_path]
                names_test = ["Loss", "F1", "wgt_path"]
                if early_stop.DataCollectorStopper(values_test, names_test, step):
                    break
        early_stop.save(log=self.LOG)        
    def Validation(self, _pred, _lab):
        _pred[_pred < 0] = 0
        _pred[_pred > 0] = 1
        _lab[_lab > 0] = 1
        return f1_score(_pred.flatten(), _lab.flatten())

def GetOptions():
    import argparse
    parser = argparse.ArgumentParser(
        description='Training Distance')
    parser.add_argument('--train_record', required=True,
                        metavar="str", type=str,
                        help='train_record')
    parser.add_argument('--test_record', required=True,
                        metavar="str", type=str,
                        help='test_record')
    parser.add_argument('--mean_file', required=True,
                        metavar="/path/to/mean_file/",
                        help='Path to the mean file to use')
    parser.add_argument('--learning_rate', required=True,
                        metavar="float", type=float,
                        help='learning rate')
    parser.add_argument('--batch_size', required=True,
                        metavar="int", type=int,
                        help='batch size')
    parser.add_argument('--epochs', required=True,
                        metavar="int", type=int,
                        help='Number of epochs to perform')    
    parser.add_argument('--weight_decay', required=True,
                        metavar="float", type=float,
                        help='Weight decay value to be applied')
    parser.add_argument('--n_features', required=True,
                        metavar="int", type=int,
                        help='Complexity of the architecture in terms of filters')
    parser.add_argument('--log', required=True,
                        metavar="folder", type=str,
                        help='Log folder to dump weights and other things')
    args = parser.parse_args()
    return args

def GetRecordSize(record):
    return len([el for el in tf.python_io.tf_record_iterator(record)])

if __name__ == '__main__':

    args = GetOptions()

    n_train = GetRecordSize(args.train_record)
    n_test  = GetRecordSize(args.test_record)

    NUMBER_OF_STEPS_FOR_ONE_EPOCH = n_train // args.batch_size
    NUMBER_OF_STEPS = NUMBER_OF_STEPS_FOR_ONE_EPOCH * args.epochs
    model = DNNDist(args.train_record,
                    args.test_record,
                    LEARNING_RATE=args.learning_rate,
                    BATCH_SIZE=args.batch_size,
                    IMAGE_SIZE=(212, 212),
                    NUM_CHANNELS=3,
                    STEPS=NUMBER_OF_STEPS,
                    LRSTEP="10epoch",
                    N_PRINT=NUMBER_OF_STEPS_FOR_ONE_EPOCH,
                    LOG=args.log,
                    WEIGHT_DECAY=args.weight_decay,
                    N_FEATURES=args.n_features,
                    N_THREADS=100,
                    MEAN_FILE=args.mean_file,
                    EARLY_STOPPING=10
                    )
    model.train(n_test)