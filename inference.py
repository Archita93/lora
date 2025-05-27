import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer
from llama_cpp import Llama
from instruction import csv_to_instruction, InstructDataset

ENTITY_TYPES = ['NAME', 'DATE', 'LOCATION', 'HOSPITAL', 'IDENTIFIER','CONTACT']  # or adapt based on your dataset


def inference(model_path, model_name, max_new_tokens, prediction_path,csv_path):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    instructions = csv_to_instruction(csv_path)
    dataset = InstructDataset(
        instructions=instructions,
        tokenizer=tokenizer,
        max_source_tokens_count=128,
        max_target_tokens_count=64,
        model_type='llama',
        only_target_loss=True,
        padding=False
    )

    model = Llama(
        model_path=model_path,
        n_gpu_layers = 35,
        n_ctx=2048,
        n_parts=1,
        use_mmap=False,
    )
    generation_config = GenerationConfig.from_pretrained(model_name)
    max_new_tokens = max_new_tokens
        
    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []
    
    for idx, instruction in enumerate(tqdm(instructions)):
        input_ids = model.tokenize(instruction['source'])
        input_ids.append(model.token_eos())
        generator = model.generate(
                input_ids,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                temp=generation_config.temperature,
                repeat_penalty=generation_config.repetition_penalty,
                reset=True,
        )

        completion_tokens = []
        for i, token in enumerate(generator):
            completion_tokens.append(token)
            if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
                break
            
        completion_tokens = model.detokenize(completion_tokens).decode("utf-8")
        extracted_list.append(extract_classes(completion_tokens), ENTITY_TYPES)
        instruction_ids.append(instruction['id'])
        target_list.append(instruction['raw_entities'])
    
    pd.DataFrame({
        'id': instruction_ids, 
        'extracted': extracted_list,
        'target': target_list
    }).to_json(prediction_path)