import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import TypedDict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import ast
from transformers import GenerationConfig, AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer  # Use AutoTokenizer for flexibility
from huggingface_hub import login
login(token='HF LOGIN')
from metric import extract_classes

ENTITY_TYPES = ['NAME', 'DATE', 'LOCATION', 'HOSPITAL', 'IDENTIFIER','CONTACT']  # or adapt based on your dataset

def csv_to_instruction(csv_path):
    df = pd.read_csv(csv_path)
    instructions = []
    for idx, row in df.iterrows():
        entities = ast.literal_eval(row["entities"])
        instructions.append({
            "instruction":  """
            You are solving the NER problem. Extract from the text words related to each 
            of the following entities: NAME, DATE, LOCATION, HOSPITAL, IDENTIFIER, CONTACT""",
        "input":row["text"],
        "output":str(entities),
        "source": f"You are solving the NER problem. Extract all named entities (NAME, DATE, LOCATION, HOSPITAL, IDENTIFIER, CONTACT) from the following text  of the following entities :\n\n{row['text']}\n\n:",
        "raw_entities": entities,
        "id": str(idx),
        })
    return instructions


class Instruction(TypedDict):
    instruction: str
    input: str
    output: str
    source: str   
    raw_entities: dict[str, list[str]]
    id: str

"""
A custom dataset to:

    Convert raw instructions into tokenized tensors (input_ids, attention_mask, labels)

    Format them for causal (e.g., LLaMA) or encoder-decoder (e.g., T5) models

__init__()

    Takes in a list of instructions and model-specific settings

    For each instruction:
        Calls convert_instruction_causal for causal LMs (e.g., LLaMA, Mistral)

        Or convert_instruction_seq2seq for encoder-decoder models (e.g., T5)

Stores all converted examples in self.processed_instructions
"""

class InstructDataset(Dataset):
    def __init__(
        self,
        instructions: list[Instruction],
        tokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        model_type: str = 'llama',
        only_target_loss: bool = True,
        padding: bool = False
    ):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.model_type = model_type
        self.only_target_loss = only_target_loss
        self.padding = padding

        self.processed_instructions = []

        for instruction in tqdm(self.instructions):
            if self.model_type in ['llama', 'mistral', 'rwkv']:
                tensors = self.convert_instruction_causal(instruction)
            elif self.model_type == 't5':
                tensors = self.convert_instruction_seq2seq(instruction)
            else:
                raise ValueError('model_type must be equals "llama", "mistral", "rwkv" or "t5"')

            self.processed_instructions.append(tensors)

    def __len__(self):
        return len(self.processed_instructions)

    def __getitem__(self, index):
        return self.processed_instructions[index]


    def convert_instruction_causal(self, instruction: dict[str, str]):
        target = instruction['output']
        source = instruction['source']        
        
        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )['input_ids']

        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)

        input_ids = source_tokens[:]
        max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2

        target_tokens = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_target_tokens_count,
            padding=False,
            truncation=True
        )['input_ids']

        input_ids += target_tokens + [self.tokenizer.eos_token_id]

        if self.padding:
            actual_length = len(input_ids)
            padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
            input_ids.extend(padding)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        if self.padding:
            labels[actual_length:] = -100
            attention_mask[actual_length:] = 0

        if self.only_target_loss:
            labels[:len(source_tokens)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


# def load_instructions(json_path):
#     return pd.read_json(json_path, orient="records").to_dict(orient="records")

def inference(model_name, instructions, max_new_tokens, prediction_path):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    dataset = InstructDataset(
        instructions=instructions,
        tokenizer=tokenizer,
        max_source_tokens_count=128,
        max_target_tokens_count=64,
        model_type='llama',  # For Qwen, this logic is still fine
        only_target_loss=True,
        padding=False
    )


    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []

    for idx, instruction in enumerate(tqdm(instructions)):
        prompt = instruction['source']
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Optionally, remove the prompt from the output if needed
        completion_text = completion_text[len(prompt):].strip()
        extracted_list.append(extract_classes(completion_text, ENTITY_TYPES))
        instruction_ids.append(instruction['id'])
        target_list.append(instruction['raw_entities'])
        sources.append(prompt)

    pd.DataFrame({
        'id': instruction_ids,
        'source': sources,
        'extracted': extracted_list,
        'target': target_list
    }).to_json(prediction_path)



def main():
    # Load tokenizer (replace with your model path or name)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

    # Prepare instructions from your CSV
    instructions = csv_to_instruction("medical_reports.csv")

    # Shuffle and split into train/test (80/20 split)
    train_instructions, test_instructions = train_test_split(
        instructions, test_size=0.2, random_state=42
    )


    inference(
    model_name="Qwen/Qwen1.5-7B-Chat",
    instructions=test_instructions,
    max_new_tokens=64,
    prediction_path="test_predictions.json")

if __name__ == "__main__":
    main()


# convert_instruction_causal()	
# Decoder-only models	
# input_ids = source + target	
# labels mask source


#   labels[:len(source_tokens)] = -100: Ignores source tokens in the loss (focuses loss on the target/output)

    # If padding is enabled, applies padding and masks labels and attention_mask appropriately
    # Returns a dictionary with:
    #     input_ids
    #     labels
    #     attention_mask




    # def convert_instruction_seq2seq(self, instruction: dict[str, str]):
    #     target = instruction['output']
    #     source = instruction['source']
        
    #     inputs = self.tokenizer(
    #         source,
    #         add_special_tokens=True,
    #         max_length=self.max_source_tokens_count,
    #         padding=False,
    #         truncation=True,
    #         return_tensors="pt"
    #     )
    #     inputs = {k: v.squeeze(0) for k, v in inputs.items()}

    #     outputs = self.tokenizer(
    #         target,
    #         add_special_tokens=True,
    #         max_length=self.max_target_tokens_count,
    #         padding=False,
    #         truncation=True,
    #         return_tensors="pt"
    #     )
    #     labels = outputs["input_ids"].squeeze(0).tolist()
    #     if labels[-1] != self.tokenizer.eos_token_id:
    #         labels.append(self.tokenizer.eos_token_id)

    #     inputs["labels"] = torch.LongTensor(labels)
    #     return inputs