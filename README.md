# Logical_MRC_ReClor

个人学习仓库，整理逻辑推理机器阅读理解的代码库。

使用RoBerta预训练模型， 使用ReClor数据集， 使用inductive learning 图神经网络

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
      
transformers = 4.1.1
模型      Roberta-pool  

100       0.364       0.354

200       0.438       0.442

300       0.548       0.556

400       0.616       0.608

500       0.636       0.644   loss达到最小

600       0.636       0.622

700       0.652       0.668

800       0.644       0.642

900       0.652       0.658

1000      0.646       0.642

1100      0.656       0.658

1200      0.654       0.636     train达到acc=0.99

1300      0.632       0.65

1400      0.650       0.654

1500      0.658       0.65

1600      0.652       0.654

1700      0.664       0.652

1800      0.666       0.642

1900      0.658       0.64

1930      0.658       0.64
    
