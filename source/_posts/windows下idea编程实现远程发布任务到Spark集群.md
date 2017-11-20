---
title: windows下idea编程实现远程发布任务到Spark集群
date: 2016-12-30 19:47:13
tags: [spark,idea,scala]
categories: spark
---
# 说明
本文的目标是：在windows下，使用idea编写spark任务，并可直接右键运行提交至远程Linux Spark集群上，不需要打包后再拷贝至远程Linux服务器上，再使用命令运行。
# 准备工作
* 软件
  * win10
  * jdk1.7(windows版本:1.7.0_79)
  * scala2.11.8(windows版本：scala-2.11.8.zip)
  * idea 2016.3.2(windows版本：ideaIU-2016.3.2.exe)
  * hadoop2.7.3(linux版本：hadoop-2.7.3.tar.gz)
  * spark2.0.2(linux版本：spark-2.0.2-bin-hadoop2.7.tgz)
  * idea scala插件（scala-intellij-bin-2016.3.4.zip，https://plugins.jetbrains.com/idea/plugin/1347-scala）
  * winutil.exe等（[winutil下载地址][1]）
  * maven3.3.9(windows版本：apache-maven-3.3.9-bin.zip)
<!--more-->
* 搭建Spark集群
  [分布式Spark集群搭建][2]
* 配置windows环境变量
  * jdk(windows版本) JAVA_HOME
  * scala(windows版本) SCALA_HOME
  * hadoop(linux版本) HADOOP_HOME
  * maven(windows版本) MAVEN_HOME
 **注意：以上环境变量均在windows下配置，括号中强调了软件包的平台版本。**
* 配置idea
  * maven配置：
     * **修改setting.xml**
     修改%MAVEN_HOME%下的conf/setting.xml为阿里云镜像
     ```
      在mirrors节点添加：
      <mirror> 
          <id>nexus-aliyun</id>
          <name>Nexus aliyun</name>
          <url>http://maven.aliyun.com/nexus/content/groups/public</url> 
          <mirrorOf>central</mirrorOf> 
      </mirror>
     ```
     * **修改idea的maven配置**
       主要是为了加快建立maven项目时的速度
        ![maven-idea-setting1][3] 
        ![maven-idea-setting2][4] 
        
 * scala pluin配置
     ![scala-idea-setting1][20] 
     ![scala-idea-setting1][21]     
        
# 开发流程        
* **新建MAVEN+SCALA项目**
    ![maven-scala1][5] 
    ![maven-scala2][6]
    ![maven-scala3][7] 
* **配置JDK、SCALA**
    ![maven-scala4][8] 
    
* **添加POM依赖**
  ```
  <properties>
    <spark.version>2.0.2</spark.version>
    <scala.version>2.11</scala.version>
  </properties>
  <dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-core_${scala.version}</artifactId>
    <version>${spark.version}</version>
  </dependency>
  <dependency>
    <groupId>org.apache.hadoop</groupId>
    <artifactId>hadoop-client</artifactId>
    <version>2.6.0</version>
  </dependency>
  ```

* **编写代码**
    ```
    import org.apache.spark.{SparkConf, SparkContext}
    import scala.math.random
    object SparkPi {
      def main(args:Array[String]):Unit = {
        val conf = new SparkConf().setAppName("Spark Pi").setMaster("spark://172.16.21.121:7077")
          .setJars(List("E:\\idea-workspace\\spark-practice\\out\\artifacts\\spark_practice_jar\\spark-practice.jar"));
    
        val spark = new SparkContext(conf)
        val slices = if (args.length > 0) args(0).toInt else 2
        val n = 100000 * slices
        val count = spark.parallelize(1 to n, slices).map { i =>
          val x = random * 2 - 1
          val y = random * 2 - 1
          if (x * x + y * y < 1) 1 else 0
        }.reduce(_ + _)
        println("Pi is roughly " + 4.0 * count / n)
        spark.stop()
      }
    } 
    ```
    其中setMaster为：spark主节点的地址。setjars为下面步骤生成的jar包在window路径下的目录

* 添加输出sparkdemo.jar
    ![artifacts1][9] 
    ![artifacts2][10] 
    ![artifacts3][11] 
    ![artifacts4][12] 

* 编译代码
    ![build][13] 

* **删除输出的sparkdemo.jar中META-INF中多余文件**
  只保留MANIFEST.MF和MAVEN文件夹
  ![delete][14] 

* **include in build勾掉**
  防止右键运行的时候，重新输出，导致mete-inf又恢复了
  ![去掉include in build][15] 

* 设置VM参数
  ![VM参数][16] 
* 右键运行
  ![run1][17] 
* 运行时可查看web控制台
  ![run2][18] 
  ![run3][19] 
  [1]: https://github.com/xuetf/spark/blob/master/idea/hadoop-common-2.2.0-bin.rar?raw=true
  [2]: /2016/12/29/Spark%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%95%99%E7%A8%8B/
  [3]: https://raw.githubusercontent.com/xuetf/spark/master/idea/maven_setting1.png
  [4]: https://raw.githubusercontent.com/xuetf/spark/master/idea/maven_setting2.png
  [5]: https://raw.githubusercontent.com/xuetf/spark/master/idea/maven_scala.png
  [6]: https://raw.githubusercontent.com/xuetf/spark/master/idea/maven_scala2.png
  [7]: https://raw.githubusercontent.com/xuetf/spark/master/idea/maven_scala3.png
  [8]: https://raw.githubusercontent.com/xuetf/spark/master/idea/maven_scala4.png
  [9]: https://raw.githubusercontent.com/xuetf/spark/master/idea/artifacts1.png
  [10]: https://raw.githubusercontent.com/xuetf/spark/master/idea/artifacts2.png
  [11]: https://raw.githubusercontent.com/xuetf/spark/master/idea/artifacts3.png
  [12]: https://raw.githubusercontent.com/xuetf/spark/master/idea/artifacts4.png
  [13]: https://raw.githubusercontent.com/xuetf/spark/master/idea/build.png
 [14]: https://raw.githubusercontent.com/xuetf/spark/master/idea/sparkdemo_mete_inf_delete.png
 [15]: https://raw.githubusercontent.com/xuetf/spark/master/idea/build2.png
[16]: https://raw.githubusercontent.com/xuetf/spark/master/idea/vm.png
[17]: https://raw.githubusercontent.com/xuetf/spark/master/idea/run-result.png
[18]: https://raw.githubusercontent.com/xuetf/spark/master/idea/spark-running.png
[19]: https://raw.githubusercontent.com/xuetf/spark/master/idea/spark-finished.png
[20]: https://raw.githubusercontent.com/xuetf/spark/master/idea/idea_scala_plugin1.png
[21]: https://raw.githubusercontent.com/xuetf/spark/master/idea/idea_scala_plugin2.png