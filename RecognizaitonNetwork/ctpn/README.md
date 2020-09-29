# text-detection-ctpn

text detection mainly based on ctpn (connectionist text proposal network). It is implemented in tensorflow. I use id card detect as an example to demonstrate the results, but it should be noticing that this model can be used in almost every horizontal scene text detection task. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/)
***
# setup
- requirements: tensorflow1.3, cython0.24, opencv-python, easydict,(recommend to install Anaconda)
- if you do not have a gpu device,follow here to [setup](https://github.com/eragonruan/text-detection-ctpn/issues/43)
- if you have a gpu device, build the library by
```shell
cd lib/utils
chmod +x make.sh
./make.sh
```
***
# parameters
there are some parameters you may need to modify according to your requirement, you can find them in ctpn/text.yml
- USE_GPU_NMS # whether to use nms implemented in cuda or not
- DETECT_MODE # H represents horizontal mode, O represents oriented mode, default is H
- checkpoints_path # the model I provided is in checkpoints/, if you train the model by yourself,it will be saved in output/
***
# demo
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ./ctpn/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. you can download it from [google drive](https://drive.google.com/open?id=0B_WmJoEtfQhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). 
- Second, prepare the training data as referred in paper, or you can download the data I prepared from previous link. Or you can prepare your own data according to the following steps. 
- Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run
```shell
cd prepare_training_data
python split_label.py
```
- it will generate the prepared data in current folder, and then run
```shell
python ToVoc.py
```
- to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/ and then run
```shell
cd ../data
ln -s TEXTVOC VOCdevkit2007
```
## train 
Simplely run
```shell
python ./ctpn/train_net.py
```
- you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
- The model I provided in checkpoints is trained on GTX1070 for 50k iters.
- If you are using cuda nms, it takes about 0.2s per iter. So it will takes about 2.5 hours to finished 50k iterations.
***
# roadmap
- [x] cython nms
- [x] cuda nms
- [x] python2/python3 compatblity
- [x] tensorflow1.3
- [x] delete useless code
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
- [ ] side refinement
***
# some results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="data/results/001.jpg" width=320 height=240 /><img src="data/results/002.jpg" width=320 height=240 />
<img src="data/results/003.jpg" width=320 height=240 /><img src="data/results/004.jpg" width=320 height=240 />
<img src="data/results/009.jpg" width=320 height=480 /><img src="data/results/010.png" width=320 height=320 />
***
## oriented text connector
- oriented text connector has been implemented, i's working, but still need futher improvement.
- left figure is the result for DETECT_MODE H, right figure for DETECT_MODE O
<img src="data/results/007.jpg" width=320 height=240 /><img src="data/results/007_oriented.jpg" width=320 height=240 />
<img src="data/results/008.jpg" width=320 height=480 /><img src="data/results/008_oriented.jpg" width=320 height=480 />
***
