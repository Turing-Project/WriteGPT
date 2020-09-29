FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

RUN apt update && apt install -y --no-install-recommends git
RUN git clone -q https://github.com/imcaspar/gpt2-ml && mkdir -p gpt2-ml/models/mega

WORKDIR /gpt2-ml

RUN perl 3rd/gdown.pl/gdown.pl https://drive.google.com/open?id=1n_5-tgPpQ1gqbyLPbP1PwiFi2eo7SWw_ models/mega/model.ckpt-100000.data-00000-of-00001
RUN wget -q --show-progress https://github.com/imcaspar/gpt2-ml/releases/download/v0.5/model.ckpt-100000.index -P models/mega
RUN wget -q --show-progress https://github.com/imcaspar/gpt2-ml/releases/download/v0.5/model.ckpt-100000.meta -P models/mega

CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root pretrained_model_demo.ipynb"]