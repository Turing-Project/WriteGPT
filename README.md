
# EssayKiller



![image](https://img.shields.io/badge/License-Apache--2.0-green) ![image](https://img.shields.io/badge/License-MIT-orange)  ![image](https://img.shields.io/badge/License-Anti--996-red)  ![image](https://img.shields.io/badge/pypi-v0.0.1a4-yellowgreen) ![image](https://img.shields.io/badge/stars-%3C%201k-blue) ![image](https://img.shields.io/badge/issues-1%20open-brightgreen)  

通用型议论文创作人工智能框架，仅限交流与科普。

Bilibili视频地址：https://www.bilibili.com/video/BV1pr4y1w7uM/

## 项目简介
EssayKiller是基于OCR、NLP领域的最新模型所构建的生成式文本创作AI框架，目前第一版finetune模型针对高考作文（主要是议论文），可以有效生成符合人类认知的文章，多数文章经过测试可以达到正常高中生及格作文水平。

| 项目作者        | 主页1           | 主页2  | 主页3 |
| ------------- |:-------------:|:----:|:---:|
| 图灵的猫       | [知乎](https://www.zhihu.com/people/dong-xi-97-29) |[B站](https://space.bilibili.com/371846699) | [Youtube](https://www.youtube.com/channel/UCoEVP6iTw5sfozUGLLWJyDg/featured) |


**致谢**

感谢开源作者[@imcaspar](https://github.com/imcaspar)  提供GPT-2中文预训练框架与数据支持。
感谢[@白小鱼博士](https://www.zhihu.com/people/youngfish42) 、[@YJango博士](https://www.zhihu.com/people/YJango) 、[@画渣花小烙](https://space.bilibili.com/402576555)、[@万物拣史](https://space.bilibili.com/328531988/) 、[@柴知道](https://space.bilibili.com/26798384/)、[@风羽酱-sdk](https://space.bilibili.com/17466521/)、[@WhatOnEarth](https://space.bilibili.com/410527811/)、[@这知识好冷](https://space.bilibili.com/403943112/)、[@科技狐](https://space.bilibili.com/40433405/) 的参与和支持
<br>

## 框架说明
- [x] 基于EAST、CRNN、Bert和GPT-2语言模型的高考作文生成AI
- [x] 支持bert tokenizer，当前版本基于clue chinese vocab
- [x] 17亿参数多模块异构深度神经网络，超2亿条预训练数据
- [x] 线上点击即用的文本生成效果demo：[17亿参数作文杀手](https://colab.research.google.com/github/EssayKillerBrain/EssayKiller_V2/blob/master/colab_online.ipynb)
- [x] 端到端生成，从试卷识别到答题卡输出一条龙服务



### Colab线上作文生成功能
国内没有足够显存的免费GPU平台，所以配合Google Drive将训练好的AI核心功能Language Network写作模块迁移到Colab。

当前线上仅开放文本生成功能，输入对应句子，AI返回生成文章。同一个句子可以输入多次，每一次输出都不同。也可以选择同时生成多篇文章。具体见：[17亿参数作文杀手](https://colab.research.google.com/github/EssayKillerBrain/EssayKiller_V2/blob/master/colab_online.ipynb)

* 第一步：安装环境
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-22-13.png)

* 第二部：加载模型
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-27-38.png)

* 第三步：文章生成
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-27-14.png)

* 写作效果
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-23-27.png)


## 本地环境
* Ubuntu 18.04.2
* Pandas 0.24.2
* Regex 2019.4.14
* h5py 2.9.0
* Numpy 1.16.2
* Tensorboard 1.15.2
* Tensorflow-gpu 1.15.2
* Requests 2.22.0
* OpenCV 3.4.2
* CUDA >= 10.0
* CuDNN >= 7.6.0

## 开发日志

* 2020.06.23 本地Git项目建立
* 2020.07.03 整体模型架构搭建，开始语料收集
* 2020.07.13 基于OCR的视觉网络训练
* 2020.08.01 GPT-2中文预训练模型微调
* 2020.08.14 Bert文本摘要模型
* 2020.08.23 通顺度判分网络测试
* 2020.09.14 排版脚本与输出装置改装

## 模型结构
整个框架分为EAST、CRNN、Bert、GPT-2、DNN 5个模块，每个模块的网络单独训练，参数相互独立。infer过程使用pipeline串联，通过外接装置直接输出到答题卡。  
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-35-00.png)


### 1. 输入
高考语文试卷作文题
>![浙江卷](https://images.shobserver.com/img/2020/7/7/37b2224ee3de441a8a040cb4f5576c2d.jpg)


### 2. 识别网络
#### 2.1 EAST文本检测
OpenCV 的EAST文本检测器是一个深度学习模型，它能够在 720p 的图像上以13帧/秒的速度实时检测任意方向的文本，并可以获得很好的文本检测精度。  
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-45-54.png)

<br>

**模型亮点**
1. 简单的管道实现在当时较高精度的文本检测。
2. 图像通过FCN处理产生像素级文本缩放地图和几何图形的多个频道。
3. 可旋转的文本框，可以检测文本也可以检测单词。

EAST文本检测器需要 OpenCV3.4.2 或更高的版本，有需要的读者可以查看 [OpenCV 安装教程](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/)。虽然EAST的模型在检测自然场景下的英文文本有着较好的性能，要实现中文场景下的中文文本检测，仍然需要重新训练模型。

**数据集处理**

中文文本识别的数据集要按照原作者的命名方式修改，即使使用ICDAR3013这类标准数据集，也需要修改对应的图片命名方式。原代码数据集的命名方式：图片1.jpg 图片1.txt。

此外，代码是通过获取文件类型然后重新命名以原来的文件类型保存的，所以文本数据和图片数据需要分开处理。

*训练命令：*
```bash
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \ --text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \ --pretrained_model_path=/tmp/resnet_v1_50.ckpt 
```


更多细节可以参考：https://zhuanlan.zhihu.com/p/64737915

<br>

*检测结果*
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-25-01.png)

除了EAST，也可以把识别网络替换为传统的CTPN等模型，github上有已经成熟的项目：https://github.com/Walleclipse/ChineseAddress_OCR

#### 2.2 CRNN文本识别

参考 https://github.com/ooooverflow/chinese-ocr 

**数据准备**

下载[训练集](https://pan.baidu.com/s/1E_1iFERWr9Ro-dmlSVY8pA)：共约364万张图片，按照99: 1划分成训练集和验证集

数据利用中文语料库（新闻 + 文言文），通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成。包含汉字、英文字母、数字和标点共5990个字符，每个样本固定10个字符，字符随机截取自语料库中的句子，图片分辨率统一为280x32。  

*修改/train/config.py中train_data_root，validation_data_root以及image_path*

**训练**
```bash
cd train  
python train.py
```

**训练结果**
```python
Epoch 3/100
25621/25621 [==============================] - 15856s 619ms/step - loss: 0.1035 - acc: 0.9816 - val_loss: 0.1060 - val_acc: 0.9823
Epoch 4/100
25621/25621 [==============================] - 15651s 611ms/step - loss: 0.0798 - acc: 0.9879 - val_loss: 0.0848 - val_acc: 0.9878
Epoch 5/100
25621/25621 [==============================] - 16510s 644ms/step - loss: 0.0732 - acc: 0.9889 - val_loss: 0.0815 - val_acc: 0.9881
Epoch 6/100
25621/25621 [==============================] - 15621s 610ms/step - loss: 0.0691 - acc: 0.9895 - val_loss: 0.0791 - val_acc: 0.9886
Epoch 7/100
25621/25621 [==============================] - 15782s 616ms/step - loss: 0.0666 - acc: 0.9899 - val_loss: 0.0787 - val_acc: 0.9887
Epoch 8/100
25621/25621 [==============================] - 15560s 607ms/step - loss: 0.0645 - acc: 0.9903 - val_loss: 0.0771 - val_acc: 0.9888
```
>![](https://github.com/ooooverflow/chinese-ocr/raw/master/demo/ocr.png)

<br>

### 2. 语言网络
#### 2.1 BERT文本摘要


BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder。模型的主要创新点在pre-train方法上，用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

模型的构成元素Transformer可以参考Google的 [Attention is all you need](https://arxiv.org/abs/1706.03762) ，BERT模型的结构如下图最左：
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-44-54.png)

对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向RNN和双向RNN的区别，直觉上来讲效果会好一些。
<br>

在原论文中，作者展示了新的语言训练模型，称为编码语言模型与下一句预测 


Original Paper : 3.3.1 Task #1: Masked LM
>Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his

规则: 会有15%的随机输入被改变，这些改变基于以下规则

* 80%的tokens会成为‘掩码’token
* 10%的tokens会称为‘随机’token
* 10%的tokens会保持不变但需要被预测

下一句预测

> Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next
Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext

规则:
* 50%的下一句会（随机）成为连续句子
* 50%的下一句会（随机）成为不关联句子

<br>

**训练**

* 哈工大的新浪微博短文本摘要[LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html)
* 教育新闻自动摘要语料[chinese_abstractive_corpus](https://github.com/wonderfulsuccess/chinese_abstractive_corpus)

```bash
python run.py --model bert
```
<br>

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-40-19.png)

测试时，需要用正则表达式过滤考试专用词，包括“阅读下面的材料，根据要求写作”，“要求：xxx”，“请完成/请结合/请综合xx”。

比如
>![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-17-17-30.png)


    人们用眼睛看他人、看世界，却无法直接看到完整的自己。所以，在人生的旅程中，我们需要寻找各种“镜子”、不断绘制“自画像”来审视自我，尝试回答“我是怎样的人”“我想过怎样的生活”“我能做些什么”“如何生活得更有意义”等重要的问题。


<br>

#### 2.2 GPT-2文本生成
![](https://github.com/prakhar21/TextAugmentation-GPT2/raw/master/gpt2-sizes.png)

参考：https://github.com/imcaspar/gpt2-ml/

预训练语料来自 [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) 以及 [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)，清洗后总文本量约 15G。
 Finetune语料来自历年满分高考作文、优质散文集以及近现代散文作品，约1000篇。  

**预训练**  
参考 [GPT2-ML](https://github.com/imcaspar/gpt2-ml/) 预训练模型，使用 [Quadro RTX 8000](https://www.nvidia.com/en-us/design-visualization/quadro/rtx-8000/) 训练 28w 步

>![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/2233.PNG)


<br>

**Finetune**

```bash
1、进入dataset目录
python pre_data.py --filepath /data/home/share1/gpt2-ml-Finetune/data-mayun_xiugai --outfile /data/home/share1/gpt2-ml-Finetune/data/22.json
filepath为finetune数据目录

2、生成tfrecord训练数据
python prepare_data.py -input_fn /data/home/share1/gpt2-ml-Finetune/data

3、finetune
CUDA_VISIBLE_DEVICES=0  python train/train_wc.py --input_file=/data/EssayKiller/gpt2-ml-Finetune/data/train.tfrecord --output_dir=/data/EssayKiller/gpt2-ml-Finetune/finetune_model --init_checkpoint=/data/EssayKiller/gpt2-ml/models/mega/model.ckpt-220000

```

<br>

### 3.判分网络

#### 3.1 DNN判分模型
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-18-59-12.png)

这部分直接调用百度API。有现成的模型就不重复造轮子了，具体实现方式百度没有开源，这里简单描述一下语言模型的概念：
语言模型是通过计算给定词组成的句子的概率，从而判断所组成的句子是否符合客观语言表达习惯。通常用于机器翻译、拼写纠错、语音识别、问答系统、词性标注、句法分析和信息检索等。  
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-18-59-57.png)

这里使用通顺度打分作为判断依据。  

#### 3.2 高考排版器

*标题*  
复用BERT_SUM生成Top3的NER粒度token作为标题

*主体*  
高考议论文的写作格式要求如下：
1. 标题居中，一般少于20字
2. 每段段首缩进两格
3. 每个字符尽量保持在字体框内
4. 字数不能过长或过短

由于模型输出的文章不保证换行和分段，通过统计高考作文的常见段数、每段句数，编写脚本对输出进行划分。大多数情况下分段排版的结果都比较合理。  
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-19-04-24.png)

<br>

## 输出
**答题卡**  
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-19-07-53.png)

**外接装置**

基于aedraw，一款开源的CNC(Computer Numerical Control数控机床)画图机器人，具有绘制图案、写字等功能，它也可以升级为激光雕刻等用途。
详细教程见 http://aelab.net/ ，不仅能自己制作一台写字绘画机器人，而且能够掌握其工作原理拓展更多的应用。  

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-19-12-07.png)

原版的输出临摹装置存在速度慢和格式不准的问题，通过改装和修改源代码得以优化

* 因为时间原因目前的手写装置还有些问题，偶尔会有漏写、越格的问题
* 视频中的作文经过后期的人工处理，补上了漏字

<br>

## 预训练模型

| 模型        | 参数量           | 下载链接  | 备注 |
| ------------- |:-------------:|:----:|:---:|
| EAST  | < 0.1 Billion  | [GoogleDrive](https://drive.google.com/file/d/1fF4IYaL7CWghYCDvRrACM57WVx83Yvny/view?usp=sharing) | 检测模型 |
| CRNN | < 0.1 Billion   | [网盘链接](https://eyun.baidu.com/s/3dEUJJg9) 提取码：vKeD| 识别模型 |
| BERT | 0.1 Billion   | [GoogleDrive](https://drive.google.com/file/d/15DbA07DZNT3gMXu2aLliA3CkuR5XHhlt/view?usp=sharing) | 摘要模型 |
| GPT-2 | 1.5 Billion   | [GoogleDrive](https://drive.google.com/file/d/1ujWYTOvRLGJX0raH-f-lPZa3-RN58ZQx/view?usp=sharing)  | 生成模型 |

整个AI的参数量分布不均匀，主要原因在于，这是一个语言类AI，99%的参数量集中在语言网络中，其中GPT-2（15亿）占88%，BERT（1.1亿）占7%，其他的识别网络和判分网络共占5%。

### 当前问题
* 输出的格式和高考作文还不能完美契合，之后的参数需要微调一下。为了国庆前完成，我还没来得及优化
* 生成的100篇作文里有很大一部分其实算不上合格的作文，有些只能勉强及格，有些甚至能拿零分（占比不多），显然GPT-2的能力有限。为了视频效果我只选了相对好的几篇做展示
* 英文版的说明还没来得及写，有空的同学可以翻译一下提个pr

## Q&A
* **我能否用EssayKiller来帮自己写作业？**  
  不能。所以有下一个问题：  
  
* **为什么缺少一些关键文件？**  
项目在一开始是完全开源的，经过慎重考虑我认为完全开源会被部分别有用心的人用以牟利，甚至用作不法用途。参考咸鱼和淘宝上一些魔改的开源框架应用。部分懂技术又不想动笔的小同志可能会让Essaykiller帮自己写作业，比如读后感、课后作文、思修小论文。我想说，这样不好。  

* **为什么不直接加密？**  
本来打算用混淆加密，但一些模块本就是开源的，所以我开源了整体的模型文件，只隐藏了关键的，包括pipeline、输入输出在内的文件，另外有些文件里也加了盐。  

* **有哪些模组可用？**  
目前完全开源，可以独立复用的部分包括：
  - [x] 检测网络
  - [x] 文本摘要网络
  - [x] 文本生成网络
  - [x] 判分网络与排版脚本  

* **为什么不用GPT-3**  
训练一个中文GPT-3的价格至少为1200万美元，折合人民币将近1亿。要是真有人训练出来一个中文GPT-3还开源模型文件了，我愿称之为最强。  

* **训练EssayKiller需要多少钱？**  
从头到尾训练完pipeline的话在1K～100K人民币不等，取决于你有无分布式集群可用  

<br>

## Citation
```
@misc{EssayKillerBrain,
  author = {Turing's Cat},
  title = {Autowritting Ai Framework},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/EssayKillerBrain/EssayKiller}},
}
```

<br>


## 参考资料  
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] ERNIE: Enhanced Representation through Knowledge Integration  
[3] Fine-tune BERT for Extractive Summarization  
[4] EAST: An Efficient and Accurate Scene Text Detector  
[5] An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition  
[6] Language Models are Unsupervised Multitask Learners  
[7] https://github.com/Morizeyao/GPT2-Chinese  
[8] https://github.com/argman/EAST  
[9] https://github.com/bgshih/crnn  
[10] https://github.com/zhiyou720/chinese_summarizer  
[11] https://zhuanlan.zhihu.com/p/64737915  
[12] https://github.com/ouyanghuiyu/chineseocr_lite  
[13] https://github.com/google-research/bert  
[14] https://github.com/rowanz/grover  
[15] https://github.com/wind91725/gpt2-ml-finetune-  
[16] https://github.com/guodongxiaren/README  
[17] https://www.jianshu.com/p/55560d3e0e8a  
[18] https://github.com/YCG09/chinese_ocr  
[19] https://github.com/xiaomaxiao/keras_ocr  
[20] https://github.com/nghuyong/ERNIE-Pytorch  
[21] https://zhuanlan.zhihu.com/p/43534801  
[22] https://blog.csdn.net/xuxunjie147/article/details/87178774/  
[23] https://github.com/JiangYanting/Pre-modern_Chinese_corpus_dataset  
[24] https://github.com/brightmart/nlp_chinese_corpus  
[25] https://github.com/SophonPlus/ChineseNlpCorpus  
[26] https://github.com/THUNLP-AIPoet/Resources  
[27] https://github.com/OYE93/Chinese-NLP-Corpus  
[28] https://github.com/CLUEbenchmark/CLUECorpus2020  
[29] https://github.com/zhiyou720/chinese_summarizer  


## 免责声明
该项目中的内容仅供技术研究与科普，不作为任何结论性依据，不提供任何商业化应用授权
