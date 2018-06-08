from SegNetsTF import UNetDist


def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH,
                    BATCH_SIZE, N_THREADS, CHANNELS=3, buffers=2000):
    dataset = tf.data.TFRecordDataset(filename_queue)
    def f_parse(x):
        return _parse_function(x, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    dataset = dataset.map(f_parse,  num_parallel_calls=N_THREADS)
    dataset = dataset.prefetch(buffer_size=buffers / 100) 
    dataset = dataset.shuffle(buffer_size=buffers)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(100)
    iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    return iterator.initializer, iterator

class DNNDist(UNetDist):
    def init_queue(self, train, test):
        """
        New queues for coordinator
        """
        with tf.device('/cpu:0'):
            self.data_init, self.train_iterator = read_and_decode(train, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          self.BATCH_SIZE,
                                                          self.N_THREADS)

            self.data_init_test, self.test_iterator = read_and_decode(test, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          1,
                                                          self.N_THREADS)
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

        self.images, self.annotation = tf.cond(self.is_training, f_true, f_false)

        if self.SUB_MEAN:
            self.images_queue = self.image - self.MEAN_ARRAY
        else:
            self.images_queue = self.image
        self.image_PH = tf.placeholder_with_default(self.images_queue, shape=[None,
                                                                              None, 
                                                                              None,
                                                                              3])
        return self.image_PH

    def train(self, tfrecord):
        
