# Relation Classification with Multi-Level Attention CNNs

### [中文Blog](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89961647)


Original paper  [Relation Classification via Multi-Level Attention CNNs（Wang/ACL2016）](http://iiis.tsinghua.edu.cn/~weblt/papers/relation-classification.pdf) with overall architecture listed below.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190508170925879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)


## Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

## Datasets
- SemEval 2010 Task 8 with 19 relations.The full introduction can be found [here](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview).


## Usage

### For Training:

```
python3 train.py 
```

### For final evaluate
 
```
python3 train.py --test 
```
```
> perl ../data/scorer.pl ../data/TEST_FILE_KEY.TXT ../data/results.txt > ../data/results_scores.txt
```

## To do
- finetune the paras to get better performance
- other todos are in the code
- any suggestions, welcome to issue

