from transformers import MT5ForConditionalGeneration, AutoTokenizer,Seq2SeqTrainingArguments,EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq,Seq2SeqTrainer

import torch
import evaluate
import numpy as np
from mbart import read_file,pair_sentences


def data_preprocessing_t5(english_file_path, chinese_file_path,device,tokenizer):
    prefix = "translate English to Chinese: "

    max_input_length = 128
    max_target_length = 128

    # Pair sentences
    paired_sentences = pair_sentences(english_file_path, chinese_file_path)

    def tokenize_data(paired_sentences, tokenizer):
        features = []

        for pair in paired_sentences:
            src_text, tgt_text = pair
            src_text = prefix + src_text

            # Tokenize the source text
            src_encoding = tokenizer(src_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

            # Tokenize the target text
            tgt_encoding = tokenizer(tgt_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

            feature = {
                "input_ids": src_encoding['input_ids'].squeeze(0),
                "attention_mask": src_encoding['attention_mask'].squeeze(0),
                "labels": tgt_encoding['input_ids'].squeeze(0)
            }
            features.append(feature)

        return features
    
    # Tokenize the data
    dataset = tokenize_data(paired_sentences,tokenizer)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])



    tokenizer.save_pretrained("model")

    return train_dataset, test_dataset, val_dataset




def train_t5(train_dataset, val_dataset,tokenizer,device):

    model_checkpoint = "google/mt5-small"
    # import model and tokenizer
    model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

   

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

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

    

    # data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    
  
    source_lang = "en"
    target_lang = "zh"

    batch_size = 16
    model_name = model_checkpoint.split("/")[-1]


    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy = "steps",
        eval_steps = 50,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        report_to = "wandb",
        predict_with_generate=True,
        load_best_model_at_end = True,
        metric_for_best_model='eval_loss',
        fp16=False,
        logging_dir='.logs',
        logging_steps=10)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    early_stopping = EarlyStoppingCallback(early_stopping_patience = 3)
    trainer.add_callback(early_stopping)

    
    trainer.train()
    eval_results = trainer.evaluate()
    # Save the model
    model.save_pretrained("model/t5")

    return eval_results,model