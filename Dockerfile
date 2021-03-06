#Whole Install Can Take 10~15 minutes   need  >1.1 G disk space
FROM ubuntu:latest
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN DEBIAN_FRONTEND=noninteractive apt install -y jupyter-notebook jupyter-core python-ipykernel
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN apt-get update && apt-get install -y python3-tk wget
RUN \
  wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
  echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list && \
  apt-get update && \
  apt-get install -y google-chrome-stable && \
rm -rf /var/lib/apt/lists/*
RUN echo jupyter nbextension enable --py --sys-prefix widgetsnbextension >> /etc/bash.bashrc
RUN echo jupyter notebook --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token=\'\' \& >> /etc/bash.bashrc
RUN echo google-chrome-stable "\"http://localhost:8888/notebooks/train.ipynb\"" --no-sandbox \& >> /etc/bash.bashrc
WORKDIR /usr/src/app
RUN pip3 install https://github.com/Anacletus/tensorflow-wheels/raw/master/v11.0/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl
COPY . .  
RUN pip3 install -r requirements.txt
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
