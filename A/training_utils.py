import re
import torch
import numpy as np
from datasets import load_dataset, load_metric
import sacrebleu
i






#clean the dataset
def clean_sentence(sentence):
    # Remove leading and trailing whitespaces
    sentence = sentence.strip()
    # Remove extra whitespaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence



def is_chinese_sentence(sentence):
    # Define a regular expression to match any character outside the Chinese Unicode ranges
    non_chinese_char_regex = re.compile('[^\u4e00-\u9fff]')
    # If the sentence has any non-Chinese characters, return False
    return not non_chinese_char_regex.search(sentence)









    



