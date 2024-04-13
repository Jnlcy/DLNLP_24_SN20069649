# README

## Machine Translation with mBART and T5
This repository contains code for training and evaluating two powerful machine translation models, mBART and T5, specifically tailored for translating text from English to Chinese. The implementation utilizes the "open_subtitles" dataset and is designed to easily switch between models using command line arguments.

### Prequisites
* python 3.9
* transformers 4.30
* Dataset from HuggingFace
* NVIDIA GPU with CUDA (Recommended for faster training)

### Installation
```
git clone https://github.com/Jnlcy/DLNLP_24_SN20069649
cd DLNLP_24_SN20069649
pip install -r requirements.txt
```

### Usage

```
python main.py [model_type]
```

Where model_type is either 'mbart' or 't5'


### Structure
* main.py: The main script that orchestrates the training and testing processes based on the input model type.
* A/mbart: Directory containing scripts specific to the mBART model (data_preprocessing_mbart, train_mbart, test_model).
* A/t5: Directory containing scripts specific to the T5 model (data_preprocessing_t5, train_t5, test_model).


### Acknowledgement

Corpus Name: OpenSubtitles
     Package: OpenSubtitles in Moses format
     Website: http://opus.nlpl.eu/OpenSubtitles-v2018.php
     Release: v2018
Release date: Tue Apr  3 23:47:52 EEST 2018

This corpus is part of OPUS - the open collection of parallel corpora
OPUS Website: http://opus.nlpl.eu

This is a new collection of translated movie subtitles from http://www.opensubtitles.org/.


