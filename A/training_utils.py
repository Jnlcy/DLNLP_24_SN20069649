import re
import numpy as np
import torch
import sacrebleu
from tqdm import tqdm








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


def test_model(model, tokenizer, test_dataset, device='cuda', max_new_tokens=50,print_example = False):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    references = []

    with torch.no_grad():  # No need to track gradients
        for item in tqdm(test_dataset, desc='Evaluating'):
            # Convert lists to tensors and move them to the specified device
            input_ids = torch.tensor(item['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(item['attention_mask']).unsqueeze(0).to(device)
            labels = torch.tensor(item['labels']).unsqueeze(0).to(device)

            # Generate translation using the model
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)

            # Decode the generated ids to text
            pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
            pred_processed = " ".join(list(pred_text))
            true_text = tokenizer.decode(labels[0], skip_special_tokens=True)

            true_processed = " ".join(list(true_text))

            #print(pred_text)
            #print(true_text)
            predictions.append(pred_processed)
            references.append([true_processed])


    if print_example:
        for i in range(5):
            print(f"Prediction: {predictions[i]}")
            print(f"Reference: {references[i]}")
            print()
    # Compute BLEU score
   

    scores = sacrebleu.corpus_bleu(predictions, references,use_effective_order=True)

    return scores


    

    


  
    






    



