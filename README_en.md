
# EssayKiller



![image](https://img.shields.io/badge/License-Apache--2.0-green) ![image](https://img.shields.io/badge/License-MIT-orange)  ![image](https://img.shields.io/badge/License-Anti--996-red)  ![image](https://img.shields.io/badge/pypi-v0.0.1a4-yellowgreen) ![image](https://img.shields.io/badge/stars-%3C%201k-blue) ![image](https://img.shields.io/badge/issues-1%20open-brightgreen)  

General AI framework for argumentation creation, for communication and popularsation only.

Video address in Bilibili: https://www.bilibili.com/video/BV1pr4y1w7uM/

## Intro
EassyKiller is a generative AI framework for article creation based on the newest models of OCR, NLP. The first version, finetune model, is for essays in College Entrance Examination (mainly argumentations). Most of the generated articles have been tested to reach the passing level of ordinary high school students.


| Author        | Homepage 1           |  Homepage 2 | Homepage 3 |
| ------------- |:-------------:|:----:|:---:|
| 图灵的猫       | [Zhihu](https://www.zhihu.com/people/dong-xi-97-29) |[Bilibili](https://space.bilibili.com/371846699) | [YouTube](https://www.youtube.com/channel/UCoEVP6iTw5sfozUGLLWJyDg/featured) |


**Special Thanks to**
Thanks to open source author [@imcaspar](https://github.com/imcaspar) for his Chinese pre-trained framework GPT-2 and data support.
Thanks to the involvement and support of [@白小鱼博士](https://www.zhihu.com/people/youngfish42), [@YJango博士](https://www.zhihu.com/people/YJango), [@画渣花小烙](https://space.bilibili.com/402576555)、[@万物拣史](https://space.bilibili.com/328531988/), [@柴知道](https://space.bilibili.com/26798384/), [@风羽酱-sdk](https://space.bilibili.com/17466521/), [@WhatOnEarth](https://space.bilibili.com/410527811/), [@这知识好冷](https://space.bilibili.com/403943112/) and [@科技狐](https://space.bilibili.com/40433405/).
<br>

## Description
- [x] AI essay generator for College Entrance Examination based on EAST, CRNN, Bert and GPT-2 language models
- [x] Supports bert tokenizer, current version based on clue chinese vocab
- [x] 1.7B-parameter multi-module heterogeneous DNN, over 200M pre-trained data
- [x] Online click-to-use demo for article generation effect: [1.7B-Parameter EssayKiller](https://colab.research.google.com/github/EssayKillerBrain/EssayKiller_V2/blob/master/colab_online.ipynb)
- [x] End-to-end generation, from problem recognition to answer sheet output



### Colab online essay generation
Since there's no free platform with enough memory GPUs, I moved the trained core function of AI, Language Network, to Colab cooperating with Google Drive

Only article generation function is available online currently. AI returns corresponding generated article as we enter sentences. Same sentence can be inputted multiple times, and every output result's different. More details go to: [1.7B-Parameter EssayKiller](https://colab.research.google.com/github/EssayKillerBrain/EssayKiller_V2/blob/master/colab_online.ipynb)

* Step 1: Set up the environment
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-22-13.png)

* Step 2: Load the model
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-27-38.png)

* Step 3: Generate articles
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-27-14.png)

* Effect
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-23-27.png)


## Local Environment
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

## Development Log

* 2020.06.23 Local git project establishment
* 2020.07.03 Set up the overall model framework and started corpus collection
* 2020.07.13 Visual network train based on OCR
* 2020.08.01 Slightly adjustment for Chinese pre-trained model GPT-2
* 2020.08.14 Bert text summarisation model
* 2020.08.23 Test for coherence scoring network
* 2020.09.14 Typesetting script and output device modifications.

## Model Structure
The whole framework is divided into 5 modules, EAST, CRNN, Bert, GPT-2 and DNN. Each module has independent training network and parameters. Infer is linked up with pipeline, and directly output to answer sheet by external devices.

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-35-00.png)


### 1. Input
Essay problem in College Entrance Examination
>![Zhejiang Province's problem](https://images.shobserver.com/img/2020/7/7/37b2224ee3de441a8a040cb4f5576c2d.jpg)


### 2. Recognition Network
#### 2.1 EAST Text Detection
EAST Text Detector by OpenCV is a deep learning model that can perform real-time detection in every direction at 13fps on a 720P image and has a good accuracy.
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-15-45-54.png)

<br>

**Highlights**
1. Simple pipelines realise high-accuracy text detection at that time.
2. Images are processed by FCN to produce pixel-level text zoom map and multiple channels of graphics.
3. Rotatable text box, can detect texts as well as words.

EAST Text Detector requires OpenCV 3.4.2 or higher. You may view [OpenCV Tutorial for Installation](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/). Although EAST model has good performance in detecting English texts in nature scene, re-training model is required to realise Chinese text detection.

**Datasets Processing**

Datasets for Chinese text detection needs renaming as the way original author did. Even if we use standard dataset like ICDAR3013, we have to edit the naming method of corresponding images. Original dataset naming method: 图片1.jpg 图片1.txt。

Besides, codes are saved by getting filetype and renaming as the original filetype. So text and image data needs processing separatedly.

*Training Command:*
```bash
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \ --text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \ --pretrained_model_path=/tmp/resnet_v1_50.ckpt 
```


More details refer to: https://zhuanlan.zhihu.com/p/64737915

<br>

*Result*

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-25-01.png)

Except EAST, we can replace recognition network with traditional models like CTPN. A mature project in GitHub: https://github.com/Walleclipse/ChineseAddress_OCR

#### 2.2 CRNN Text Recognition

Reference: https://github.com/ooooverflow/chinese-ocr 

**Getting Ready**

Download [Training Set](https://pan.baidu.com/s/1E_1iFERWr9Ro-dmlSVY8pA), about 3.64M images in total. Divided into training set and validation set according to 99:1

The data is randomly generated using Chinese corpuses (news + classical Chinese writing) through changes in font, size, grayscale, blur, perspective, and stretching. There are 5990 characters in total including Chinese and English characters, numbers and punctuations. Length of each sample is fixed at 10 characters. Characters are extracted randomly from corpus library. The image resolution is unified at 280x32.

*edit train_data_root, validation_data_root and image_path in /train/config.py*

**Train**
```bash
cd train  
python train.py
```

**Result**
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

### 2. Language Network
#### 2.1 BERT Text Summarisation

BERT, short for Bidirectional Encoder Representation from Transformers, which is the Encoder of two-way Transformer. Model's main innovation is the pre-train method, using Masked LM and Next Sentence Prediction to capture representation of word and sentence's level. 

For an element in the model, Transformer, you can refer to [Attention is all you need](https://arxiv.org/abs/1706.03762) by Google. BERT's model struture as the left of the image below
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-44-54.png)

Comparing with OpenAI GPT, BERT has the two-way Transformer block connection. Just like the difference between one-way RNN and two-way, the effect looks better on intuition.
<br>

In the paper, authors shows the new language model training methods, which are "masked language model" and "predict next sentence".


Original Paper : 3.3.1 Task #1: Masked LM
>Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his

Rules: Randomly 15% of input token will be changed into something, based on under sub-rules


* Randomly 80% of tokens, gonna be a `[MASK]` token
* Randomly 10% of tokens, gonna be a `[RANDOM]` token
* Randomly 10% of tokens, will be remain as same. But need to be predicted.

Next Sentence Prediction

> Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next
Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext

Rules:
* Randomly 50% of next sentence, gonna be continuous sentence.
* Randomly 50% of next sentence, gonna be unrelated sentence.

<br>

**Train**

* HIT's Sina Blog short text summarisation: [LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html)
* Educational news auto summarisation corpus: [chinese_abstractive_corpus](https://github.com/wonderfulsuccess/chinese_abstractive_corpus)

```bash
python run.py --model bert
```
<br>

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-16-40-19.png)

While doing tests, regular expression is required to filter proper words for exams. Including "阅读下面的材料，根据要求写作", "要求：xxx" and "请完成/请结合/请综合xx".

Example
>![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-17-17-30.png)


    人们用眼睛看他人、看世界，却无法直接看到完整的自己。所以，在人生的旅程中，我们需要寻找各种“镜子”、不断绘制“自画像”来审视自我，尝试回答“我是怎样的人”“我想过怎样的生活”“我能做些什么”“如何生活得更有意义”等重要的问题。


<br>

#### 2.2 GPT-2 Text Generation
![](https://github.com/prakhar21/TextAugmentation-GPT2/raw/master/gpt2-sizes.png)

Reference: https://github.com/imcaspar/gpt2-ml/

Pre-trained corpuses are from [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) and [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus). Size of texts after cleaning is about 15G.
 Finetune corpuses are from high-scoring essays in College Entrance Examination in past years, high-quality prose collections and modern prose. About 1000 articles in total.

**Pre-training**  
Refer to [GPT2-ML](https://github.com/imcaspar/gpt2-ml/) pre-trained model, using [Quadro RTX 8000](https://www.nvidia.com/en-us/design-visualization/quadro/rtx-8000/) to train for over 280K steps.

>![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/2233.PNG)


<br>

**Finetune**

```bash
1. Enter the dataset directory
python pre_data.py --filepath /data/home/share1/gpt2-ml-Finetune/data-mayun_xiugai --outfile /data/home/share1/gpt2-ml-Finetune/data/22.json
filepath is finetune's path

2. Generate tfrecord training data
python prepare_data.py -input_fn /data/home/share1/gpt2-ml-Finetune/data

3. finetune
CUDA_VISIBLE_DEVICES=0  python train/train_wc.py --input_file=/data/EssayKiller/gpt2-ml-Finetune/data/train.tfrecord --output_dir=/data/EssayKiller/gpt2-ml-Finetune/finetune_model --init_checkpoint=/data/EssayKiller/gpt2-ml/models/mega/model.ckpt-220000

```

<br>

### 3. Scoring Network

#### 3.1 DNN Scoring Network
![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-18-59-12.png)

Directly calling Baidu's API. "Don't Repeat Yourself" since there's a ready-to-use model. Baidu hasn't open sourced the specific implementation, here is a brief description of it:
Language model judges whether the composed sentence conforms to objective language expression habits by calculating the probability of composing a sentence of a given word. Generally used for machine translator, speech correction, voice recognition, Q&A system, part-of-speech labeling, syntactic analysis, info retrieval, etc.

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-18-59-57.png)

Coherence is used here as the basis for judgement.


#### 3.2 Typesetter for College Entrance Examination

*title*  
Reuse BERT_SUM to generate Top 3 NER granularity token as title

*body*  
Article format requirements in College Entrance Examination:
1. Title is centred, commonly less that 20 characters
2. Each paragraph is indented by 2 characters
3. Each character tries to keep in the square box
4. Word count cannot be too long or to short

Since articles outputted by the model cannot guarantee wrapping and paragraph breaking, we analysed common count of paragraphs in an article and sentences in a paragraph, and wrote a script to divide the output article. Most of them are reasonably divided.

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-19-04-24.png)

<br>

## Output
**Answer Sheet**  

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-19-07-53.png)

**Exteral Device**

Based on aedraw, an open source CNC (Computer Numerical Control) drawing robot. It can draw patterns, do handwriting. Also can be upgraded for laser engraving.
Tutorial go to http://aelab.net/ , you can not only make a drawing & writing robot but masters its working principle and expands something more as well.

![](https://github.com/EssayKillerBrain/EssayKiller_V2/blob/master/References/attachments/Clipboard_2020-09-29-19-12-07.png)

The original output copying device has problems with slow speed and inaccurate format, can be optimised by modifications and editing source code.

* There are some issues with the handwriting device. Omission and overstepping may occur occasionally.
* Essays in my video were manually processed to fix the missing characters.

<br>

## Pre-trained Models

| Model        |  Amount of Parameter          |  Download Link | Remarks |
| ------------- |:-------------:|:----:|:---:|
| EAST  | < 0.1 Billion  | [Google Drive](https://drive.google.com/file/d/1fF4IYaL7CWghYCDvRrACM57WVx83Yvny/view?usp=sharing) | Detection Model |
| CRNN | < 0.1 Billion   | [cloud drive](https://eyun.baidu.com/s/3dEUJJg9) Extraction Code: vKeD| Recognition Model |
| BERT | 0.1 Billion   | [Google Drive](https://drive.google.com/file/d/15DbA07DZNT3gMXu2aLliA3CkuR5XHhlt/view?usp=sharing) | Summarisation Model |
| GPT-2 | 1.5 Billion   | [Google Drive](https://drive.google.com/file/d/1ujWYTOvRLGJX0raH-f-lPZa3-RN58ZQx/view?usp=sharing)  | Generation Model |

The whole AI's parameter amount is unevenly distributed. Mainly reason is that it's a language AI, 99% of the parameters focus on the language network. GPT-2 (1.5B parameters) accounts for 88%, BERT (110M parameters) accounts for 7% and other recognition network and scoring network account for 5%.

### Current Issues
* Output format and College Entrance Examination's are not perfect. The following parameters require slight adjustment. I haven't got time to optimise so as to finish before National Day.
* Most of the 100 generated essays are actually not passing. Some barely reach the passing level. Some even got zero point (not much). Obviously GPT-2 has a limited ability. For vidio effect, I only chose a few good ones.

## Q&A
* **Can I use EssayKiller to help me do homework?**  
No you can't. So there's the following question:
  
* **Why some key files are missing?**  
The project is completely open source at first. But after careful consideration, I think fully open source may cause it used by people for profit, even for illegal use. Refer to some magically modified open source frameworks in Xianyu and Taobao. And someone proficient in the model and lazy to write essays by himself might use it help do his homework, like some little essays. I'd like to say that that's no good.

* **Why not encrypt directly?**  
I orginally intended to use obfuscated encryption, but some modules are already open source. Therefore I made whole model files open source, and only some key ones including pipeline, input & output files are hidden. What's more some files are added salts.

* **Which modules are available to use?**  
Now it's fully open source, parts able to independently reused includes:
  - [x] Detection Network
  - [x] Text Summarisation Network
  - [x] Text Generation Network
  - [x] Scoring Network and typesetting scripts

* **Why not use GPT-3?**  
Price of training a Chinese GPT-3 is at least $12M, converting into CNY is nearly ¥100M. If someone has trained a Chinese GPT-3 and open sourced the module files, I would call him the BADASS.

* **How much does it cost to train the EssayKiller?**  
Training pipeline from beginning to end costs CNY 1K ~ 100K, depends on whether you have distributed cluster to use or not.

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


## References 
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


## Disclaimer
Contents in the project are for research and science popularsation purpose only, not as any conclusion basis. I do not offer any authorisation for commercial applications.
