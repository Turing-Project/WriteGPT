## Data Preprocess

* Standford Core NLP Toolkit

https://stanfordnlp.github.io/CoreNLP/

`set classpath C:\Users\VY\Desktop\Project\chinese_summarizer\utils\stanford-corenlp-full-2017-06-09\stanford-corenlp-3.8.0.jar`

`python preprocess.py`

## Train

`python train.py`

## Test
* make data

`python preprocess.py -oov_test True -data_name DATA_NAME -raw_path RAW_PATH`