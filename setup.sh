FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

RUN apt update && apt install -y --no-install-recommends git

!pip install pandas==0.24.2 &> tmp.log
!pip install regex==2019.4.14 &> tmp.log
!pip install h5py==2.10.0 &> tmp.log
!pip install numpy==1.18.4 &> tmp.log
!pip install tensorboard==1.15.0 &> tmp.log
!pip install tensorflow-gpu==1.15.2 &> tmp.log
!pip install tensorflow-estimator==1.15.1 &> tmp.log
!pip install tqdm==4.31.1 &> tmp.log
!pip install requests==2.22.0 &> tmp.log
!pip install ujson==2.0.3 &> tmp.log

!mkdir -p /home/EssayKiller/AutoWritter/finetune/trained_models

%cd /home/EssayKiller/AutoWritter/finetune/
!perl /home/EssayKiller/AutoWritter/scripts/gdown.pl https://drive.google.com/open?id=1ujWYTOvRLGJX0raH-f-lPZa3-RN58ZQx trained_models/model.ckpt-280000.data-00000-of-00001
!wget -q --show-progress https://github.com/EssayKillerBrain/EssayKiller/releases/download/v1.0/model.ckpt-280000.index -P /home/EssayKiller/AutoWritter/finetune/models/mega
!wget -q --show-progress https://github.com/EssayKillerBrain/EssayKiller/releases/download/v1.0/model.ckpt-280000.meta -P /home/EssayKiller/AutoWritter/finetune/models/mega


CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root pretrained_model_demo.ipynb"]