from transformers import MBartForConditionalGeneration, Seq2SeqTrainer, DataCollatorForSeq2Seq,MBart50Tokenizer,Seq2SeqTrainingArguments
import torch
import numpy as np
import sacrebleu
from training_utils import clean_sentence,is_chinese_sentence



def data_preprocessing_mbart(dataset, tokenizer):

   
    source_lang = "en"
    target_lang = "zh_cn"

    def preprocess_function(examples):
        inputs = [example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    #shuffle the dataset
    dataset = dataset['train'].shuffle(seed=42)

   #use the first 100000 pairs for now
    dataset = dataset.select(range(500000))

    #filter out non-Chinese sentences
    dataset = dataset.filter(lambda example: is_chinese_sentence(example['translation']['zh_cn']))

    print(dataset['translation'][:5])


    tokenized_dataset = dataset.map(preprocess_function, batched=True,remove_columns=dataset.column_names)

    

    #split the dataset

    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, test_size, val_size])
    #print the lengths of the datasets
    print(len(train_dataset), len(test_dataset), len(val_dataset))

    tokenizer.save_pretrained("model")

    return train_dataset, test_dataset, val_dataset

  




def train_mbart(train_dataset, val_dataset,tokenizer,device):
    #initialize tokenizer
    model_checkpoint = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint)
    print("Tokenizer loaded")

    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

    print("number of parameters:", model.num_parameters())


    
    

    # Define the compute_metrics function
    def compute_metrics(eval_preds):


        preds, labels = eval_preds
        preds = preds[0] if isinstance(preds, tuple) else preds

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Ensure character splitting for Chinese evaluation in both predictions and references
        decoded_preds = [" ".join(list(pred)) for pred in decoded_preds]
        bleu_references = [[" ".join(list(label))] for label in decoded_labels]  # Correct format for sacrebleu references


        # Calculate BLEU scores
        bleu_scores = sacrebleu.corpus_bleu(decoded_preds, bleu_references, force=True, use_effective_order=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        avg_gen_len = np.mean(prediction_lens)

        # Combine and format results
        result = {

            "bleu": bleu_scores.score,
            "gen_len": avg_gen_len,
        }

        return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    


    source_lang = "en"
    target_lang = "zh_cn"

    batch_size = 4
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy = "epoch",
        save_strategy ="epoch",
        warmup_steps = 500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        learning_rate=5e-5,
        lr_scheduler_type='linear',
        num_train_epochs=2,
        report_to = "wandb",
        predict_with_generate=True,
        load_best_model_at_end = True,
        metric_for_best_model="rougeL",
        fp16=False,
        logging_dir='.logs',
        logging_steps=100)


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("model/mbart")

    # Evaluate the model
    eval_results = trainer.evaluate()
   

    return eval_results["bleu"],model

















