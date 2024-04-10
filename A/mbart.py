from transformers import MBartForConditionalGeneration, Trainer, DataCollatorForSeq2Seq,EarlyStoppingCallback,Seq2SeqTrainingArguments
import torch
import numpy as np
import evaluate
from sacrebleu.metrics import BLEU




def read_file(source_path, target_path):
    with open(source_path, 'r') as file:
        source_lines = file.readlines()
    source_lines = [line.strip() for line in source_lines if line.strip()]
    #filter out duplicates
    #source_lines = list(set(source_lines))


    with open(target_path, 'r') as file:
        target_lines = file.readlines()
    target_lines = [line.strip() for line in target_lines if line.strip()]
    #filter out duplicates
    #target_lines = list(set(target_lines))

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

    #shuffle the pairs
    np.random.seed(42)
    np.random.shuffle(paired_sentences)

    #use the first 10000 pairs for now
    paired_sentences = paired_sentences[:100000]

    return paired_sentences


def tokenize_data(paired_sentences, tokenizer):
    features = []
    
    for pair in paired_sentences:
        src_text, tgt_text = pair
        src_text = tokenizer.bos_token + '<en_XX> ' + src_text
        tgt_text = tokenizer.bos_token + '<zh_CN> ' + tgt_text

        # Tokenize the source text
        src_encoding = tokenizer(src_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

        # Tokenize the target text
        tgt_encoding = tokenizer(tgt_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

        feature = {
            "input_ids": src_encoding['input_ids'].squeeze(0),  # Remove batch dimension
            "attention_mask": src_encoding['attention_mask'].squeeze(0),  # Remove batch dimension
            "labels": tgt_encoding['input_ids'].squeeze(0)  # Remove batch dimension
        }
        features.append(feature)
    
    return features

def data_preprocessing_mbart(english_file_path, chinese_file_path,tokenizer):

    

    # Pair sentences
    paired_sentences = pair_sentences(english_file_path, chinese_file_path)

    # Tokenize the data
    dataset = tokenize_data(paired_sentences,tokenizer)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])


    tokenizer.save_pretrained("model")

    return train_dataset, test_dataset, val_dataset

  




def train_mbart(train_dataset, val_dataset,tokenizer,device):
    # Define the post-processing function
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    

    # Define the compute_metrics function
    def compute_metrics(eval_preds):

        metric = evaluate.load_metric("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    # Load the model
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
    model.to(device)
    print("Model loaded")
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args =Seq2SeqTrainingArguments(
        output_dir='./results',          # output directory
        evaluation_strategy = 'steps',
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        eval_steps = 10,
        save_total_limit=3,
        no_cuda=False,                  # use GPU
        fp16=True,                      # Use mixed precision
        load_best_model_at_end = True,
        metric_for_best_model='eval_loss',
        predict_with_generate=True
    )

    # Trainer
    trainer = Trainer(
        model=model,                         #  Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset ,            # evaluation dataset
        data_collator=data_collator ,       # Data collator
        compute_metrics=compute_metrics
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience = 3)
    trainer.add_callback(early_stopping)

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("model/mbart")

    # Evaluate the model
    eval_results = test_model(model, tokenizer, val_dataset, device)
   

    return eval_results,model

def test_model(model, tokenizer, test_dataset, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    references = []
    
    with torch.no_grad():  # No need to track gradients
        for item in test_dataset:
            # Move tensors to the correct device
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(device)
            labels = item['labels'].unsqueeze(0).to(device)
            
            # Generate translation using the model
            output = model.generate(input_ids, attention_mask=attention_mask)
            
            # Decode the generated ids to text
            pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
            true_text = tokenizer.decode(labels[0], skip_special_tokens=True)
            
            predictions.append(pred_text)
            references.append([true_text])  # BLEU expects a list of possible references
    
    # Compute BLEU score
    bleu = BLEU()
    scores = bleu.corpus_score(predictions, references)
    
    return scores


    

    


  
    















