from diffusion_lm.model import CodeFusion
import wandb
from datasets import load_dataset
from transformers import RobertaTokenizer


DATASET = "bigcode/the-stack-smol-xs"  # for testing
# DATASET = "bigcode/the-stack-smol"  # for training

if __name__ == "__main__":
    wandb.login()
    run = wandb.init(
        project="codefusion-replication",
        config={
            "learning_rate": 5e-4,  #5e-4 is the lr from the paper
            "epochs": 4,
        })
    data = load_dataset(DATASET, "python")
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    sample1 = data['train'][0]['content']
    tokenized_text = tokenizer(sample1).input_ids
