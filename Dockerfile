FROM pytorch/pytorch

LABEL author='Patrik Bey, <patrik.bey@ucl.ac.uk>'
LABEL version="0.1"
LABEL description="This container contains the baseline model for deep lesion deficit mapping introduced in Pombo et al. (2023)"
LABEL project="Deep variational lesion-deficit mapping; Pombo et al. (2023)"
LABEL reference="http://arxiv.org/abs/2305.17478"

COPY Docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt


RUN apt-get update
RUN apt-get install figlet

RUN mkdir -p /templates
RUN mkdir "/src"

COPY Docker/data "/templates"
ENV TEMPLATEDIR="/templates"

COPY Docker/code "/src"
ENV SRCDIR="/src"

WORKDIR "/src"


