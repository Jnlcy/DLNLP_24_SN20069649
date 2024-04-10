import torch
from transformers import MBart50Tokenizer
from A.mbart import data_preprocessing_mbart, train_mbart,test_model
from A.t5 import train_t5,data_preprocessing_t5
import gc






# ======================================================================================================================
# Paths to dataset files
english_file_path = "Datasets/OpenSubtitles.en-zh_cn.en"
chinese_file_path = "Datasets/OpenSubtitles.en-zh_cn.zh_cn"

# initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#======================================================================================================================
# mbart
#initialize tokenizer

tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
print("Tokenizer loaded")

# Data preprocessing
train_dataset, val_dataset, test_dataset = data_preprocessing_mbart(english_file_path, chinese_file_path,device,tokenizer)
# ======================================================================================================================

#train model

bleu_mbart_val,model= train_mbart(train_dataset, val_dataset,tokenizer,device) # Train model based on the training set (you should fine-tune your model based on validation set.)
bleu_mbart_test = test_model(model, tokenizer, test_dataset, device='cuda')

# ======================================================================================================================
model = None
optimizer = None

del model, train_dataset, val_dataset, test_dataset
gc.collect()  # Explicitly calls garbage collection        

torch.cuda.empty_cache()  # Clear cache memory



# ======================================================================================================================
# t5
train_dataset, val_dataset, test_dataset = data_preprocessing_t5(english_file_path, chinese_file_path,device,tokenizer)
bleu_t5_val,model= train_t5(train_dataset, val_dataset,tokenizer,device) # Train model based on the training set (you should fine-tune your model based on validation set.)
bleu_t5_test= test_model(model, tokenizer, test_dataset, device='cuda')



model = None
optimizer = None
del model, train_dataset, val_dataset, test_dataset
gc.collect()  # Explicitly calls garbage collection        

torch.cuda.empty_cache()  # Clear cache memory




# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};TB:{},{};'.format(bleu_mbart_val, bleu_mbart_val, bleu_t5_test, bleu_t5_test))
     

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'