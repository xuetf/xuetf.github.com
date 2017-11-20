---
title: redis分布式环境搭建教程
date: 2016-12-29 15:56:43
categories: redis
tags: [redis, 缓存, 分布式, HA方案, 环境]
---
# redis部署说明

* **版本**  
  使用redis最新版3.2.3进行安装
* **主从关系**  
  使用1个主节点，3个从节点。主节点提供读写操作，从节点只提供读操作。主节点Master安装在dbp模块，提供大量的写操作服务；  3个从节点。
* **哨兵机制**  
  配置3个哨兵，主节点dbp安装1个哨兵，另外3台从服务器选其中两台各安装一个。作为HA高可用方案，防止主节点单点失败，通过重新选举主节点实现故障快速转移。
<!--more-->

# 安装具体步骤
* 解压
* 安装gcc
* 进入redis的bin目录，先执行 make MALLOC=libc； 再执行make install
* 配置文件：先拷贝redis目录下的配置文件redis.conf和sentinel.conf到/usr/local/etc(或其他任意目录)，再修改
    * **redis节点配置**
        <code>**bind 主机ip**                        #主机ip
        **protected-mode no**                     #保护模式关闭，否则不能通过远程连接，哨兵机制也不起作用，下面使用密码进行安全保证
        **port 端口**                          #端口
        **daemonize yes**                       #守护进程
        **pidfile  /var/run/redis_端口.pid**            #进程号，命名规则redis_端口号.pid
        logfile /usr/local/logs/redis/redis_端口.log   #日志文件
        **dir  /usr/local/data/redis/端口**          #持久化文件夹，必须是空文件夹
        **requirepass 密码**    #认证密码
        **masterauth 密码**    #和认证密码一致
        **maxmemory 最大内存**  #eg:10g
        **maxmemory-policy**        allkeys-lru   #lru算法回收
        </code>
    * **从节点需要额外配置**
        <code>slaveof 主机 ip  #例如slaveof  172.16.21.127  6379</code>
    * **Sentinel哨兵节点**
    <code>port  端口    #命名规则： 本机redis端口前加个2,比如redis:6379: 则sentinel：26379
        **sentinel announce-ip**  主机ip
        **protected-mode  no**  #需要手动添加这条。
        **dir**  /usr/local/data/sentinel_端口    #空文件夹
        **logfile**  /usr/local/logs/redis/sentinel_端口.log
        **sentinel monitor 主节点名称 主节点ip 主节点端口 仲裁至少需要的哨兵数** #eg：sentinel monitor mymaster  172.16.21.127 6379 2
        **sentinel auth-pass 主节点名称 密码**   #认证
        </code>
* **进入redis的src目录启动redis和sentinel**
    <code>**reids-server redis配置文件** 
    #eg:redis-server /usr/local/etc/redis_6379.conf 
    **redis-sentinel sentinel配置文件** &
    #eg:redis-sentinel /usr/local/etc/sentinel_26379.conf &
    </code>
* **依次启动主节点和从节点后，使用redis-cli连接**
    <code>**reids-cli -h ip地址 -p 端口 -a 密码**
   **sentinel reset mymaster** #重置哨兵状态*
    使用命令查看部署情况，info replication可查看集群状态
    </code>


# 具体配置参见
<https://github.com/xuetf/redis>

# 参考
<http://blog.csdn.net/ownfire/article/details/51546543>
<http://www.ilanni.com/?p=11838>