# Train Custom Spacy NER (token-classifier) models

The main goal of this repository is to make easy the training of custom NER model.

![](./data/spacy_ner_pipeline.png)

# Table of content

- [0. About](#0-about)
- [1. Installation](#1-installation)
- [2. GSome details](#2-some-details)
- [3. To do](#3-to-do)

## 0. About
The main goal of this repository is to make easy the training of custom NER model. Before using it, you need to have label data in *.jsonl* format. You can use Doccano (https://doccano.github.io/doccano/) for instance to label your data. The code is constructed in a way, you only need to update the config file *config.yaml* and launch *main.py* to train and evaluate a spacy model. The cleaning and data preprocessing are taking care of. 
Customs NER models can be very useful in social science or in linguistic. in our case, we use it for 4 purposes :
- detect different value of pronoun ("on" in french)
- detect different value of present time (historical present, generic present, enunciation present)
- detect the lexical field associated to death (explicit and implicit mentions)
- detect the lexical field associated with the body, sensation and perception

## 1. Installation
To make it work: (works on 3.9 also)
- Create a virtual environment (python3.8 -m venv /path/to/env)
- Activate the environment (source /path/to/env/bin/activate)
- Clone the directory (git clone )
- Install dependencies (pip install -r requirements.txt)
- Change the config file (*config.yaml*) according to the data file
- Launch the training: *python3.8 main.py*

## 2. Usage

### 2.1. Config file
The config file is a yaml file. It contains all the information needed to train a spacy model. The config file is divided in 2 parts:
1)  data: contains the path to the data file and the path to the output directory
For each ner model you want to train, you need to add a section to the data part. For instance, if you want to train a model to detect the lexical field associated with death, you need to add the following section:
```
data_death :
    origin_file : "death.jsonl"
    validation_file : ""
    labels:  ['DEATH_IMPLICITE', 'DEATH_EXPLICITE']
    dataset_folder : "./data/data_death"
```


2)  training: contains the parameters for the training, the path to the config folder and a section for each specific model you want to train. For instance, if you want to train a model to detect the lexical field associated with death, you need to add the following section:
```
  model_death:
    gpu : 0
    config_name : "death_config_transformers.cfg"
    model_folder : "./models/model_death"
    performance_filename : "performances_death.json"
```


 
