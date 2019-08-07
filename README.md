# DuReader Dataset  
[TOC]
DuReader is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:  
DuReader 是中文中新的大型真实世界和人工来源 MRC 数据集。DuReader 专注于现实世界的开放问题解答。DuReader 相对于现有数据集的优势总结如下:  

 - Real question  
 - 真实的问题
 - Real article  
 - 真实的文章  
 - Real answer  
 - 真实的回答
 - Real application scenario  
 - 实际应用程序方案  
 - Rich annotation  
 - 丰富的注释  

# DuReader Baseline Systems
DuReader system implements 2 classic reading comprehension models([BiDAF](https://arxiv.org/abs/1611.01603) and [Match-LSTM](https://arxiv.org/abs/1608.07905)) on [DuReader dataset](https://ai.baidu.com//broad/subordinate?dataset=dureader). The system is implemented with 2 frameworks: [PaddlePaddle](http://paddlepaddle.org) and [TensorFlow](https://www.tensorflow.org).  
DuReader系统在【DuReader数据集】的基础上实现了2个经典阅读理解模型【BiDAF】和【Match-LSTM】。该系统通过两个框架实现：Paddlepaddle和TensorFlow。  

## How to Run  
## 如何运行  
### Download the Dataset  
###　下载数据集　　
To Download DuReader dataset:  
要下载数据集：

```
cd data && bash download.sh
```
For more details about DuReader dataset please refer to [DuReader Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader).  
如果想了解关于DuReader数据集更多的细节，请参考[DuReader Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader)。  

### Download Thirdparty Dependencies  
### 下载第三方依赖库  
We use Bleu and Rouge as evaluation metrics, the calculation of these metrics relies on the scoring scripts under "https://github.com/tylin/coco-caption", to download them, run:  
我们使用Blue和Rouge作为评估指标，这些指标的计算依赖于"https://github.com/tylin/cococaption" 。下载他们之后，然后运行：  

```bash
cd utils && bash download_thirdparty.sh
```

### Preprocess the Data  
### 预处理数据  
After the dataset is downloaded, there is still some work to do to run the baseline systems. DuReader dataset offers rich amount of documents for every user question, the documents are too long for popular RC models to cope with. In our baseline models, we preprocess the train set and development set data by selecting the paragraph that is most related to the answer string, while for inferring(no available golden answer), we select the paragraph that is most related to the question string. The preprocessing strategy is implemented in `utils/preprocess.py`. To preprocess the raw data, you should first segment 'question', 'title', 'paragraphs' and then store the segemented result into 'segmented_question', 'segmented_title', 'segmented_paragraphs' like the downloaded preprocessed data, then run:  
下载数据集之后，仍然要做一些准备工作才能运行基线系统。DuReader数据集为每个用户的问题提供了丰富的documents，但是这些documents过长，对流行的
RC模型来讲难以处理。在我们的基线模型中，我们通过选择与答案最相关的段落来预处理训练集和开发集数据，而对于推断（没有可用的黄金答案），我们选择与问题最相关的段落。预处理的策略由*'utils/preprocess.py'*实现。为了预处理原始数据，你应该先切割*'question'*、*'title'*、*'paragraphs'*字符串，并把切割后的结果存储在*'segemented_question'*、*'segemented_title'*、*'segemented_paragraphs'*，结果就像已经下载的preprocessed文件夹中的数据一样。切割完字符串之后，然后要运行语句：  

```
cat data/raw/trainset/search.train.json | python utils/preprocess.py > data/preprocessed/trainset/search.train.json
```
The preprocessed data can be automatically downloaded by `data/download.sh`, and is stored in `data/preprocessed`, the raw data before preprocessing is under `data/raw`.  
已经预处理的数据可以通过*'data/download.sh'*自动下载，它会被存储到*'data/preprocessed'*这个文件夹中，原始数据（未处理的数据）存储在*'data/raw'*文件夹中。  

### Run PaddlePaddle 运行PaddlePaddle  

We implement a BiDAF model with PaddlePaddle. Note that we have an update on the PaddlePaddle baseline (Feb 25, 2019). The major updates have been noted in `paddle/UPDATES.md`. On the dataset of DuReader, the PaddlePaddle baseline has better performance than our Tensorflow baseline. 
我们使用PaddlePaddle框架实现一个BiDAF模型。请注意，基于PaddlePaddle框架的基线模型已经在2019年2月25号进行了更新。主要的更新可以在*'paddle/UPDATES.md'*中看到。基于DuReader数据集，PaddlePaddle基线模型比TensorFlow基线模型有更好的表现。在PaddlePaddle基线中也支持多gpu训练。  
The PaddlePaddle baseline includes the following procedures: paragraph extraction, vocabulary preparation, training, evaluation and inference. All these procedures have been wrapped in `paddle/run.sh`. You can start one procedure by running run.sh with specific arguments. The basic usage is:  
PaddlePaddle基线包括以下的过程：段落提取（paragraph extraction）、词汇准备（vocabulary preparation），训练（training）,评估（evaluation）和推理（inference）。所有这些过程都包装在*'paddle/run.sh'*中。您可以通过运行具有特定参数run.sh来启动一个过程。基本用法是：  

```
sh run.sh --PROCESS_NAME --OTHER_ARGS
```

PROCESS_NAME can be one of `para_extraction`, `prepare`, `train`, `evaluate` and `predict` (see the below detailed description for each procedure). OTHER_ARGS are the specific arguments, which can be found in `paddle/args.py`. 
PROCESS_NAME可以是'para_extraction'、 'prepare'、'train'、'evaluate'、'predict'中的任何一个(看下方的各个过程的具体描述)。OTHER_ARGS是特定的参数，这些可以在*'paddle/args.py'*中找到。  

In the examples below (except for 'Paragraph Extraction'), we use the demo dataset (under `data/demo`) by default to demonstrate the usages of `paddle/run.sh`.   
在下方的例子中（不包括'Paragraph Extraction'），我们默认用了demo数据集（在'data/demo'数据集下）来演示*'paddlepaddle/run.sh'*的用法。  

#### Environment Requirements 环境要求  
Please note that we only tested the baseline on PaddlePaddle v1.2 (Fluid) with Python 2.7.13. To install PaddlePaddle, please see [PaddlePaddle Homepage](http://paddlepaddle.org) for more information.
请注意，我们只在PaddlePaddle v1.2(Fluid)、python2.7.13环境下测试过。如果要安装PaddlePaddle,请参考[PaddlePaddle Homepage](http://paddlepaddle.org)获取更多信息。  

#### Paragraph Extraction 段落提取  
We incorporate a new strategy of paragraph extraction to improve the model performance. The details have been noted in `paddle/UPDATES.md`. Please run the following command to apply the new strategy of paragraph extraction on each document:
我们采用了段落提取的新策略，以提高模型性能。详情已在*'paddle/UPDATES.md'*中注明。请运行以下命令，以便在每个文档上应用段落提取的新策略：  
```
sh run.sh --para_extraction
```

Note that the full preprocessed dataset should be ready before running this command (see the "Preprocess the Data" section above). The results of paragraph extraction will be saved in `data/extracted/`. This procedure is only required before running the full dataset, if you just want to try vocabulary preparation/training/evaluating/inference with demo data, you can sikp this one.  
请注意，在运行这条命令之前，完整的预处理数据集应准备就绪。（看上方"PreProcess the Data"部分）。段落提取的结果会保存在*'data/extracted/'*文件夹中。这个过程仅在运行完整数据集之前是必须的，如果你只想尝试使用demo数据进行词汇词汇准备/训练/评估/推理，你可以跳过这一部分。  
#### Vocabulary Preparation 词汇准备  

Before training the model, you need to prepare the vocabulary for the dataset and create the folders that will be used for storing the models and the results. You can run the following command for the preparation:
在训练模型之前,您需要为数据集准备词汇表,并创建将用于存储模型和结果的文件夹。您可以运行以下命令进行准备:  
```
sh run.sh --prepare
```
The above command uses the data in `data/demo/` by default. To change the data folder, you need to specify the following arguments:  
默认情况下,上述命令使用"data/demo/"中的数据。要更改数据文件夹,需要指定以下参数:

```
sh run.sh --prepare --trainset ../data/extracted/trainset/zhidao.train.json ../data/extracted/trainset/search.train.json --devset ../data/extracted/devset/zhidao.dev.json ../data/extracted/devset/search.dev.json --testset ../data/extracted/testset/zhidao.test.json ../data/extracted/testset/search.test.json
```

#### Training

To train a model (on the demo trainset), please run the following command:

```
sh run.sh --train --pass_num 5
```
This will start the training process with 5 epochs. The trained model will be evaluated automatically after each epoch, and a folder named by the epoch ID will be created under the folder `data/models`, in which the model parameters are saved. If you need to change the default hyper-parameters, e.g. initial learning rate and hidden size, please run the commands with the specific arguments. 

```
sh run.sh --train --pass_num 5 --learning_rate 0.00001 --hidden_size 100
```

More arguments can be found in `paddle/args.py`.


#### Evaluate
To evaluate a specific model (on the demo devset), please run the following command:

```
sh run.sh --evaluate  --load_dir YOUR_MODEL_DIR
```
The model under `YOUR_MODEL_DIR` (e.g. `../data/models/1`) will be loaded and evaluated.

#### Inference (Prediction)
To do inference (on the demo testset) by using a trained model, please run: 

```
sh run.sh --predict  --load_dir YOUR_MODEL_DIR 
```
The predicted answers will be saved in the folder `data/results`.

#### The performance of PaddlePaddle Baseline on DuReader 2.0
|      Model     | Dev ROUGE-L | Test ROUGE-L |
| :------------- | :---------: | :----------: |
| before update  |    39.29    |     45.90    |
| after update   |    47.68    |     54.66    |

The results in the above table are obtained by using 4 P40 GPU cards with batch size = 4*32. If using a single card with a smaller batch size (e.g. 32), the performance might be slightly lower, but should be higher than ROUGE-L=47 on the devset. 

**Note**: for convinience, we also provide the trained model parameters which can be used for inference directly. To reproduce the resutls in the table, please download the [model parameters and vocabulary files](https://nlpc-du.cdn.bcebos.com/reading/baidu-2019-mrc-paddle-baseline.tar
) first, and follow the steps in the "Paragraph Extraction", "Evaluate" and "Inference" section above. 


#### Submit the test results
Once you train a model that is tuned on the dev set, we highly recommend you submit the predictions on test set to the site of DuReader for evaluation purpose. To get inference file on test set:

1. make sure the training is over.
2. select the best model under `data/models` according to the training log.
3. predict the results on test set.
4. [submit the prediction result file](http://ai.baidu.com/broad/submission?dataset=dureader).

### Run Tensorflow

We also implements the BIDAF and Match-LSTM models based on Tensorflow 1.0. You can refer to the [official guide](https://www.tensorflow.org/versions/r1.0/install/) for the installation of Tensorflow. The complete options for running our Tensorflow program can be accessed by using `python run.py -h`. Here we demonstrate a typical workflow as follows: 

#### Preparation
Before training the model, we have to make sure that the data is ready. For preparation, we will check the data files, make directories and extract a vocabulary for later use. You can run the following command to do this with a specified task name:

```
python run.py --prepare
```
You can specify the files for train/dev/test by setting the `train_files`/`dev_files`/`test_files`. By default, we use the data in `data/demo/`

#### Training
To train the reading comprehension model, you can specify the model type by using `--algo [BIDAF|MLSTM]` and you can also set the hyper-parameters such as the learning rate by using `--learning_rate NUM`. For example, to train a BIDAF model for 10 epochs, you can run:

```
python run.py --train --algo BIDAF --epochs 10
```

The training process includes an evaluation on the dev set after each training epoch. By default, the model with the least Bleu-4 score on the dev set will be saved.

#### Evaluation
To conduct a single evaluation on the dev set with the the model already trained, you can run the following command:

```
python run.py --evaluate --algo BIDAF
```

#### Prediction
You can also predict answers for the samples in some files using the following command:

```
python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json 
```

By default, the results are saved at `../data/results/` folder. You can change this by specifying `--result_dir DIR_PATH`.

## Run baseline systems on multilingual datasets

To help evaluate the system performance on multilingual datasets, we provide scripts to convert MS MARCO V2 data from its format to DuReader format. 

[MS MARCO](http://www.msmarco.org/dataset.aspx) (Microsoft Machine Reading Comprehension) is an English dataset focused on machine reading comprehension and question answering. The design of MS MARCO and DuReader is similar. It is worthwhile examining the MRC systems on both Chinese (DuReader) and English (MS MARCO) datasets. 

You can download MS MARCO V2 data, and run the following scripts to convert the data from MS MARCO V2 format to DuReader format. Then, you can run and evaluate our DuReader baselines or your DuReader systems on MS MARCO data. 

```
./run_marco2dureader_preprocess.sh ../data/marco/train_v2.1.json ../data/marco/train_v2.1_dureaderformat.json
./run_marco2dureader_preprocess.sh ../data/marco/dev_v2.1.json ../data/marco/dev_v2.1_dureaderformat.json
```

## Copyright and License
Copyright 2017 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
