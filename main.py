import torch
import gc
from datasets import load_dataset
from transformers import MBart50Tokenizer,AutoTokenizer
from A.mbart import data_preprocessing_mbart, train_mbart
from A.t5 import train_t5, data_preprocessing_t5
from A.training_utils import test_model

def main(model_type):
    # Load the dataset
    dataset = load_dataset("open_subtitles", lang1="en", lang2="zh_cn")
    print(dataset['train'][:5])

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and train the model
    if model_type == 'mbart':
        tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
        # Tokenizer for mBART initialized
        print("Tokenizer for mBART loaded")
        train_dataset, val_dataset, test_dataset = data_preprocessing_mbart(dataset, tokenizer)
        bleu_val, model = train_mbart(train_dataset, val_dataset, tokenizer, device)
        bleu_test = test_model(model, tokenizer, test_dataset, device)

    elif model_type == 't5':
        # Tokenizer for T5 initialized
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")  
        print("Tokenizer for T5 loaded")
        train_dataset, val_dataset, test_dataset = data_preprocessing_t5(dataset, tokenizer)
        bleu_val, model = train_t5(train_dataset, val_dataset, tokenizer, device)
        bleu_test = test_model(model, tokenizer, test_dataset, device)

    else:
        raise ValueError("Unsupported model type. Choose 'mbart' or 't5'.")

    # Cleanup
    del model, train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Output results
    print(f'BLEU scores for {model_type} - Validation: {bleu_val}, Test: {bleu_test}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate a translation model.")
    parser.add_argument('model_type', type=str, choices=['mbart', 't5'],
                        help='Model type to train and evaluate (mbart or t5).')
    args = parser.parse_args()

    main(args.model_type)
