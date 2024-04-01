from transformers import MBart50Tokenizer, MBartForConditionalGeneration, Trainer, TrainingArguments
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import os

# Set the device
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Paths to dataset files
english_file_path = "Datasets/OpenSubtitles.en-zh_cn.en"
chinese_file_path = "Datasets/OpenSubtitles.en-zh_cn.zh_cn"



def read_file(source_path, target_path):
    with open(source_path, 'r') as file:
        source_lines = file.readlines()
    source_lines = [line.strip() for line in source_lines if line.strip()]
    #filter out duplicates
    source_lines = list(set(source_lines))


    with open(target_path, 'r') as file:
        target_lines = file.readlines()
    target_lines = [line.strip() for line in target_lines if line.strip()]
    #filter out duplicates
    target_lines = list(set(target_lines))

    return source_lines, target_lines

def pair_sentences(english_file_path, chinese_file_path):
    # Check if both files have the same number of lines
    english_lines, chinese_lines = read_file(english_file_path, chinese_file_path)
    assert len(english_lines) == len(chinese_lines), "Files are not aligned"

    # Pair the sentences together
    paired_sentences = list(zip(english_lines, chinese_lines))
    # Now `paired_sentences` is a list of tuples, where each tuple is a pair of corresponding English and Chinese sentences

    #print the first 5 pairs
    print(paired_sentences[:5])

    #use the first 3000 pairs for now
    paired_sentences = paired_sentences[:3000]

    return paired_sentences


def tokenize_data(paired_sentences):
     # Initialize lists to store tokenized input ids and attention masks, and labels (for the target texts)
    input_ids = []
    attention_masks = []
    labels = []

    src_lang_code = tokenizer.lang_code_to_id["en_XX"]  # English
    tgt_lang_code = tokenizer.lang_code_to_id["zh_CN"]  # Chinese

    # Load tokenizer
    tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
    print("Tokenizer loaded")

    for pair in paired_sentences:
        # MBart expects the language code at the beginning of the text
        src_text, tgt_text = pair
        src_text = tokenizer.bos_token + src_lang_code + ' ' + src_text
        tgt_text = tokenizer.bos_token + tgt_lang_code + ' ' + tgt_text

        # Tokenize the source text
        src_encoding = tokenizer(src_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

        # Tokenize the target text
        tgt_encoding = tokenizer(tgt_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

        # Add the encoded texts to the lists
        input_ids.append(src_encoding['input_ids'].squeeze(0))  # Remove batch dimension
        attention_masks.append(src_encoding['attention_mask'].squeeze(0))  # Remove batch dimension
        labels.append(tgt_encoding['input_ids'].squeeze(0))  # Remove batch dimension
    
    # Convert the lists to tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)


    # Create a PyTorch dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
  

    return tokenizer,dataset


def data_preprocessing(english_file_path, chinese_file_path):

    # Pair sentences
    paired_sentences = pair_sentences(english_file_path, chinese_file_path)

    # Tokenize the data
    tokenizer, dataset = tokenize_data(paired_sentences)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])



    tokenizer.save_pretrained("model")

    return train_dataset, test_dataset, val_dataset

    





def train_model(train_dataset, val_dataset):
    # Load the model
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
    model.to(device)
    print("Model loaded")

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        no_cuda=False,                  # use GPU
        fp16=True,                      # Use mixed precision
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

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    return model


def test_model(model, test_dataset, tokenizer, device):
    model.eval()  # Set the model to evaluation mode
    
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=16)  # Adjust batch_size according to your needs and hardware capabilities
    
    predictions = []
    references = []
    
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # Decode the generated ids to strings
        batch_predictions = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in outputs]
        predictions.extend(batch_predictions)
        
        # Assuming labels are already in the batch (and are token ids)
        batch_references = [tokenizer.decode(label_id, skip_special_tokens=True) for label_id in batch['labels']]
        references.extend([[ref] for ref in batch_references])  # Note: Each reference wrapped in a list for corpus_bleu
    
    # Compute BLEU score
    smooth_fn = SmoothingFunction().method1 
    bleu_score = corpus_bleu(references, [pred.split() for pred in predictions], smoothing_function=smooth_fn)
    print(f"BLEU score: {bleu_score * 100:.2f}")
    
    # Save predictions to a file
    save_path = "predictions.txt"
    with open(save_path, "w") as file:
        for prediction in predictions:
            file.write(prediction + "\n")
    print(f"Predictions saved to {save_path}")
    
    return predictions

#process the data
train_dataset, test_dataset, val_dataset= data_preprocessing(english_file_path, chinese_file_path)

# Train the model
model = train_model(train_dataset, val_dataset)


# Test the model
predictions = test_model(model, test_dataset)






