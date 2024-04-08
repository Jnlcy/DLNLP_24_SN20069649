import torch
from transformers import MBart50Tokenizer
from A.mbart import data_preprocessing, train_model,evaluate_model
import gc






# ======================================================================================================================
# Paths to dataset files
english_file_path = "Datasets/OpenSubtitles.en-zh_cn.en"
chinese_file_path = "Datasets/OpenSubtitles.en-zh_cn.zh_cn"

# initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#initialize tokenizer

tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
print("Tokenizer loaded")

# ======================================================================================================================
# Data preprocessing
train_dataset, val_dataset, test_dataset = data_preprocessing(english_file_path, chinese_file_path,device,tokenizer)
# ======================================================================================================================

# Task A

model,bleu_A_train,bleu_A_val = train_model(train_dataset, val_dataset,tokenizer,device) # Train model based on the training set (you should fine-tune your model based on validation set.)
bleu_A_test = evaluate_model(model, test_dataset,tokenizer,device)   # Test model based on the test set.

model = None
optimizer = None
del model, train_dataset, val_dataset, test_dataset
gc.collect()  # Explicitly calls garbage collection        

torch.cuda.empty_cache()  # Clear cache memory



'''# ======================================================================================================================
# Task B
model_B = B(args...)
acc_B_train = model_B.train(args...)
acc_B_test = model_B.test(args...)
Clean up memory/GPU etc...
'''




# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};TB:{},{};'.format(bleu_A_train, bleu_A_test))
      #,
         #                                               acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'