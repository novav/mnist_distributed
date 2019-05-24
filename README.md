mnist_distributed
## 采用异步训练方式

```
1、创建集群 ClusterSpec & Server
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,  job_name=FLAGS.job_name, task_index=FLAGS.task_index)

2、设置ps节点
    tf.train.replica_device_setter(cluster=cluster)
    server.join()

3、设置worker节点
    chief 设置    
    
4、同步训练配置【可选】
    同步
        train.SyncReplicasOptimizer
    同步&chief
        chief_queue_runner 
5、train——session
    # sv =tf.train.Supervisor
    # sess = sv.prepare_or_wait_for_session(server.target)
    tf.train.MonitoredTrainingSession()
    # tf.train.Supervisor已经被弃用了
```
