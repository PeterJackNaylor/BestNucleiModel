import tensorflow as tf
import numpy as np


sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

def compute_mean_op(list_rec):
    count = 0
    image_sum = tf.zeros([212, 212, 3], dtype=tf.float64)
    for rec in list_rec:
        for example in tf.python_io.tf_record_iterator(rec):
            features = tf.parse_single_example(example, features={'image_raw':tf.FixedLenFeature([], tf.string)})
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.cast(image, tf.float64)
            image = tf.reshape(image, [396, 396, 3])
            image_sum += image[92:-92, 92:-92, :]
            count += 1
    return count, image_sum



def GetOptions():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute mean')
    parser.add_argument('records', metavar='records', type=str, nargs='+',
                    help='list of records')
    parser.add_argument('records', metavar='records', type=str, nargs='+',
                    help='list of records')
    parser.add_argument('--output', required=True,
                        metavar="string",
                        help='Output name for the mean')    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = GetOptions()
    total, image_cumsum = compute_mean_op(args.records)
    res = sess.run(image_cumsum)
    res = np.mean(res / total, axis=(0,1))
    np.save(args.output, res)