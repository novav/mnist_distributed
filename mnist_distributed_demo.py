"""
 author：simon.coding@gmail.com
 time：2019/5/24 16:20
 tools：PyCharm
 采用异步训练方式

"""
# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.app.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', './tmp/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 10, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '192.168.x.9:9900,192.168.x.8:9930', 'Comma-separated list of hostname:port pairs')
# flags.DEFINE_string('ps_hosts', '192.168.x.9:9900', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '192.168.x.9:9901,192.168.x.8:9931','Comma-separated list of hostname:port pairs')

# flags.DEFINE_string('ps_hosts', '127.0.0.1:9900', 'Comma-separated list of hostname:port pairs')
# flags.DEFINE_string('worker_hosts', '127.0.0.1:9901,127.0.0.1:9931','Comma-separated list of hostname:port pairs')

# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')

FLAGS = flags.FLAGS


def main(unused_argv):
    print('--------------> data -->', FLAGS.data_dir)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    print('+++++++++++++++++++++++++++++++++++')
    print(ps_spec)
    print(worker_spec)
    print('+++++++++++++++++++++++++++++++++++')

    # S1 创建集群
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # S2-ps
    if FLAGS.job_name == 'ps':
        server.join()

    # S2-worker
    is_chief = (FLAGS.task_index == 0)
    # tf.train.replica_device_setter    # 设置参数到ps
    worker_device = "/job:worker/task:%d" % (FLAGS.task_index)
    # with tf.device(tf.train.replica_device_setter(cluster=cluster, worker_device=worker_device, # ps_device='/job:ps/cpu:0')):
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                  cluster=cluster)):

        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

        x, y, y_ = build_net()

        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, 'summary')
        summaries.append(tf.summary.scalar('learning_rate', FLAGS.learning_rate))

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        summary_op = tf.summary.merge(summaries)
        
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        # sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1, global_step=global_step)
        # sess = sv.prepare_or_wait_for_session(server.target)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)

    config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
    ) #device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    scaffold = tf.train.Scaffold(init_op=init_op,
                                     summary_op=summary_op)

    sess = tf.train.MonitoredTrainingSession(
            config=config,
            master=server.target,
            is_chief=is_chief,
            scaffold=scaffold,
            summary_dir=train_dir,
            save_checkpoint_steps=10000,
            save_summaries_steps=1000,
        )

    print('Worker %d: Session initialization  complete.' % FLAGS.task_index)

    time_begin = time.time()
    print('Traing begins @ %f' % time_begin)

    local_step = 0
    while True:
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        _, step = sess.run([train_step, global_step], feed_dict=train_feed)
        local_step += 1

        if local_step % 1000 == 0:
            now = time.time()
            print('%f: Worker %d: traing step %d dome (global step:%d)' % (now, FLAGS.task_index, local_step, step))

        if step >= FLAGS.train_steps:
            break

    time_end = time.time()
    print('Training ends @ %f' % time_end)
    train_time = time_end - time_begin
    print('Training elapsed time:%f s' % train_time)

    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print('After %d training step(s), validation cross entropy = %g' % (FLAGS.train_steps, val_xent))
    sess.close()


def build_net():

    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                            stddev=1.0 / IMAGE_PIXELS), name='hid_w')
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')
    sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                           stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
    sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)
    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    return x, y, y_

"""
U9
nohup python3 -u mnist_distributed_demo.py --job_name=ps --task_index=0 > 0ps_001.log 2>&1 &
nohup python3 -u mnist_distributed_demo.py --job_name=worker --task_index=0  --data_dir=logs/minist > 0w_001.log 2>&1 &
U8
nohup python3 -u mnist_distributed_demo.py --job_name=ps --task_index=1  > 0ps_001.log 2>&1 &
nohup python3 -u mnist_distributed_demo.py --job_name=worker --task_index=1 --data_dir=logs/minist > 0w_001.log 2>&1 &
"""
if __name__ == '__main__':
    print('=============================')
    tf.app.run()
