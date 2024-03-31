import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import argparse


def main(model_save_path, lr):
    # model_name = "google/gemma-7b-it"
    # exp_name = 'phi2/exp_1_lr_5e-4'

    data_path = 'public_10k_unique_rewrite_prompt.csv'
    model_path = "microsoft/phi-2"
    output_path = f'checkpoint/{model_save_path.split("/")[-1]}'

    epochs=5
    batch_size=1 # 2 
    max_seq_length=512 # 1024 
    # model_save_path =  f'{exp_name}_adapter'
    # lr = 1e-4


    df = pd.read_csv(data_path)
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=False,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                quantization_config=bnb_config,
                                                trust_remote_code=True,
                                                use_auth_token=True)
    model.config.gradient_checkpointing = False


    def token_len(text):
        tokenized = tokenizer(text, return_length=True)
        length = tokenized['length'][0]
        return length

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['rewritten_text'])):
            ori_text = example['original_text'][i]
            rew_text = example['rewritten_text'][i]
            rew_prompt = example['rewrite_prompt'][i]
            text = f"Instruct: Original Text:{ori_text}\nRewritten Text:{rew_text}\nWrite a prompt that was likely given to the LLM to rewrite original text into rewritten text.Output: {rew_prompt}"
            if token_len(text) > max_seq_length:
                continue
            output_texts.append(text)
        return output_texts


    response_template = "Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                            tokenizer=tokenizer)


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "dense"],
    )

    args = TrainingArguments(
        output_dir = output_path,
        fp16=True,
        learning_rate=lr,
        optim="adafactor",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        gradient_accumulation_steps=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        )

    trainer = SFTTrainer(
        model=model,
        args = args,
        max_seq_length=max_seq_length,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    pd.DataFrame(trainer.state.log_history).to_csv(f'{model_save_path}/log_history.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--save_path', type=str, help='Adapter save directory')
    parser.add_argument('--lr', type=float, help='Learning Rate')
    args = parser.parse_args()
    # lr = 1e-4
    # exp_name = 'phi2/exp_1_lr_5e-4'

    main(args.save_path, args.lr)
