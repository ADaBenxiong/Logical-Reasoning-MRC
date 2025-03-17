# Verb-driven machine reading comprehension with dual-graph neural network

Our logical reasoning machine reading comprehension code repository.

We present a verb-driven dual-graph network(**VDGN**) that utilizes core verbs of sentences to model the inter-sentence relationship by the ability of verbs to express linguistic context and the shortest dependency path to model the relationship between entities of intra-sentence.

We verify our results on ReClor dataset and LogiQA dataset.


Our code reference:

[https://github.com/yuweihao/reclor](https://github.com/yuweihao/reclor%EF%BC%8C)  
[https://github.com/Eleanor-H/DAGN](https://github.com/Eleanor-H/DAGN)
##   directory structure
- reclor data
	- train.json
	- val.json
	- test.json
- roberta-large
	- config.json
	- vocab.json
	- pytorch_model.bin

## How to run
	
	1. install dependencies
	pip install -r requirements.txt
	
	2. train model
	bash run_roberta_large.sh


## Results

Our experimental results on the **ReClor** dataset
|       Model         |Dev|Test|
|----------------|-------------------------------|-----------------------------|
|RoBerta|62.80            |55.60           |
|EIGN|65.60            |60.00           |
|EIGN+FGM|67.00|61.00|

The leaderboard of ReClor dataset is : [Leaderboard - EvalAI](https://eval.ai/web/challenges/challenge-page/503/leaderboard)
  

We have experimented many times and the best test has been achieved 61.70%


Our experimental results on the **LoigQA** dataset
|       Model         |Dev|Test|
|----------------|-------------------------------|-----------------------------|
|RoBerta|35.02|35.33|
|EIGN|37.48|38.56|
|EIGN+FGM|38.86|38.86|

We use 1 RTX 2080Ti GPU for all experiments.   

The experimental results will be slightly different in different experimental environments

## Future

We open source our code, anyone interested could have a try.

If our work is helpful to you, hope you can click a star to show encouragement.

Our repository will continue to be updated to get better results on more datasets. And we will continue to  research the task of logical reasoning.
