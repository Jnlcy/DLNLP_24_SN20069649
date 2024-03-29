from transformers import MBart50Tokenizer, MBartForConditionalGeneration, Trainer, TrainingArguments
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
import numpy as np
import os




class Seq2SeqDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Paths to your Moses files
english_file_path = "Datasets/OpenSubtitles.en-zh_cn.en"
chinese_file_path = "Datasets/OpenSubtitles.en-zh_cn.zh_cn"

# Load the files
print("Loading Source Language File")
with open(english_file_path, 'r') as file:
    english_lines = file.readlines()

print("Loading Target Language File")
with open(chinese_file_path, 'r') as file:
    chinese_lines = file.readlines()

# Check if both files have the same number of lines
assert len(english_lines) == len(chinese_lines), "Files are not aligned"

# Pair the sentences together
paired_sentences = list(zip(english_lines, chinese_lines))




# Load tokenizer
tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
print("Tokenizer loaded")



# Now `paired_sentences` is a list of tuples, where each tuple is a pair of corresponding English and Chinese sentences

#clean data, delete empty lines and duplicates
paired_sentences = [(en, zh) for en, zh in paired_sentences if en.strip() and zh.strip()]
paired_sentences = list(set(paired_sentences))
print(paired_sentences[0:5])


#First test on small dataset
paired_sentences = paired_sentences[:1000]


# Tokenize the data
tokenized_data = tokenizer.prepare_seq2seq_batch(src_texts=[pair[0] for pair in paired_sentences], tgt_texts=[pair[1] for pair in paired_sentences], return_tensors="pt")

# print example
print(tokenized_data['input_ids'][0])
print(tokenized_data['labels'][0])

# Create a PyTorch dataset
dataset = Seq2SeqDataset(tokenized_data)

# Split the dataset
train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

# Create a PyTorch dataloader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


#Load the model
model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
model.to(device)
print("Model loaded")

# Training arguments(using cpu for now, will switch to gpu later)
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,                         #  Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Train the model
trainer.train()





# Save the model

model.save_pretrained("model")
tokenizer.save_pretrained("model")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")


# Generate some text
for i in range(5):
    input = tokenizer("translate English to Chinese: Hello, my name is John", return_tensors="pt")
    input.to(device)
    output = model.generate(**input)
    translated_text = tokenizer.batch_decode(output, skip_special_tokens=True)




