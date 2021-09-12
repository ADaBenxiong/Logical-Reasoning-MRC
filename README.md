# Logic-Driven Machine Reading Comprehension with Graph Convolutional Networks

整理逻辑推理机器阅读理解代码库。
使用RoBerta预训练模型，使用inductive learning图神经网络
在ReClor数据集上、LogiQA数据集上进行实验。

代码参考https://github.com/yuweihao/reclor， https://github.com/Eleanor-H/DAGN

在目录下需要创建保存数据集以及保存模型的文件夹

目录结构如下：

  --reclor_data
  
    --train.json
    
    --val.json
      
    --test.json
      
  --roberta-large
  
    --config.json
      
    --vocab.json
      
    --pytorch_model.bin
    
