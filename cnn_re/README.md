# Relation Classification with CNNs

### [中文Blog](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89877420)


Original paper [Relation Classification via Convolutional Deep Neural Network](https://www.aclweb.org/anthology/C14-1220) and [Relation Extraction: Perspective from Convolutional Neural Networks](https://cs.nyu.edu/~thien/pubs/vector15.pdf) with overall architecture listed below.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190506101633612.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190507170007403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)

Thanks to @FrankWork original [code](https://github.com/FrankWork/conv_relation)  and @Thidtc in [here](https://github.com/Thidtc/Relation-Classification-via-Convolutional-Deep-Neural-Network)

## Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

## Datasets
- SemEval 2010 Task 8 with 19 relations.The full introduction can be found [here](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview).


## Usage

### * For Training:

1. Prepare original SemEval 2010 data in `data` folder , you can download in [this site](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50)

2. Convert original data into desired format with`clean.py` file in `datapro` folder

3. Training, models will be save at saved_models/
```
#python3 train.py --num_epochs=150 --word_dim=50
```


### * For Inference
1. Also first preprosess with the original test data like `training` ;
2. Run the code, and a `result.txt` file with be saved in `data` folder;
```
#python3 train.py --num_epochs=150 --word_dim=50
```
### For final evaluate
 The SemEval 2010 Task 8 has its written scorer for  final evaluate, run, and final f1 score will be written into `results_scores.txt`
```
> perl ../data/scorer.pl ../data/TEST_FILE_KEY.TXT ../data/results.txt > ../data/results_scores.txt
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190508095920452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)

**<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = 81.30% >>>**

## To do
- add hypernyms to the lexical feature
- finetune the paras to get better performance
- other todos are in the code
- any suggestions, welcome to issue
