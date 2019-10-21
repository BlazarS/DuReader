# args.py中的参数  
#### --prepare    
执行prepare过程的参数，创建目录，准备词汇表和词嵌入。  
词嵌入其实就是将数据的原始表示表示成模型可处理的或者是更dense的低维表示。  要想更深入了解词嵌入的含义，轻百度一下。  
#### --train
执行train过程的参数，训练模型。  
#### --evaluate  
执行evaluate过程的参数，在开发集上评估模型  
#### --predict  
执行predict过程的参数，用已经训练好的模型推断测试机的答案  
#### embed_size  
嵌入表的维度，数据类型是**int**,默认是300。  只在--prepare出现。
#### --hidden_size  
运行隐藏单位的大小，数据类型是**int**，默认是300  。在train,evaluate,predict中出现。
#### --learning_rate  
学习率的大小，类型是**float**，默认是0.001  只在--train中出现。
#### --optim  
优化器类型，类型是**string**，默认是'adam'  。只在--train中出现。
#### --weight_decay  
权重衰减，类型是**float**，默认是0.0001  。只在--train中出现。
#### --drop_rate  
类型是**float**，默认是0.0  
#### --random_seed  
类型是**int**，默认是123  
#### --batch_size  
mini_batch批处理数据的数据号，类型是**int**，默认是32  
#### --pass_num  
要训练的次数，类型是**int**,默认是5  
#### --use_gpu  
是否使用gpu，类型是distutils.util.strtobool，默认是True  
#### --log_interval  
记录每n批数据的训练损失，类型是**int**，默认是50  
#### --max_p_num  
类型是**int**,默认是5  。--prepare --train --evaluate --predict都用到了这一参数  
#### --max_a_len  
类型是**int**,默认是200  。--prepare --train --evaluate --predict都用到了这一参数  
#### --max_p_len  
类型是**int**,默认是500  。--prepare --train --evaluate --predict都用到了这一参数
#### --max_q_len  
类型是**int**,默认是60  。--prepare --train --evaluate --predict都用到了这一参数
#### --doc_num  
类型是**int**,默认是5  。run.py无这一参数。
#### --vocab_dir  
词汇表的存储位置，默认是../data/vovab  
#### --save_dir  
模型保存的文件夹，默认是../data/models  
#### save_interval  
保存间隔，每n passes保存训练的模型，类型是**int**,默认是1  
#### --load_dir  
指定加载训练模型的路径，类型是**str**，无默认值  
#### --log_path  
日志路径，如果不设置，log会在屏幕上输出  
#### --result_dir  
结果文件夹，默认值是../data/results/  
#### result_name  
保存的结果的文件的名称，默认值是test_result  
#### --trainset 
加载train数据集文件，默认是../data/demo/trainset/search.train.json，默认值表示使用demo文件夹中的train数据集  
#### --devset  
加载dev数据集文件，默认是../data/demo/devset/search.dev.json，默认值表示使用demo文件夹中的dev数据集  
#### --testset  
加载test数据集文件，默认是../data/demo/testset/search.test.json，默认值表示使用demo文件夹中的trian数据集  
#### --enable  
如果设置,则使用连续评估日志运行任务，无默认值  
#### --para_print  
打印出来debug信息  
#### --dev_interval  
评估每n批的开发集损耗，默认是-1

