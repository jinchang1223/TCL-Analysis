import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
import xgrammar
from xgrammar import GrammarCompiler, GrammarMatcher
from xgrammar.tokenizer_info import TokenizerInfo
import json
from transformers_gad.generation.xgrammar_decoding_recorder import XGrammarDecodingRecorder
from pathlib import Path

# TODO:
# 1. 确认输出是否一样，把Glaiveai2K跑完
# 2. 跑更难的数据集，如果有fail的情况，观察常出现的错误token进行分类

NUM_ITER = 1
# MODEL_ID = "TinyLlama/TinyLlama_v1.1"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
GRAMMAR_PATH = "data/Glaiveai2K/"
HISTORY_PATH = "tries/Glaiveai2K/"
OUTPUT_PATH = "tries/Glaiveai2K_output.json"
DEVICE = "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

def main():
    device = torch.device(DEVICE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.to(dtype=DTYPE)
    model.resize_token_embeddings(len(tokenizer))
    config = AutoConfig.from_pretrained(MODEL_ID)
    
    schema_files = list(Path(GRAMMAR_PATH).glob("*.json"))
    
    # # Filter schema files that don't exist in history path
    # schema_files_to_process = [f for f in schema_files if f.name not in history_file_names]
    
    # print(f"Found {len(schema_files_to_process)} new schema files to process")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    idx = 0
    for schema_file in tqdm(schema_files, desc="Processing schemas"):
        # print(f"Processing {schema_file.name}...")
    
        # Load JSON schema
        with open(GRAMMAR_PATH + schema_file.name, "r") as file:
            schema = json.load(file)

        # # Load EBNF grammar
        # with open(GRAMMAR_PATH, "r") as file:
        #     grammar_str = file.read()

        # Initialize logits processor for the grammar
        # tokenizer_info = TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
        # compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        # compiled_grammar = compiler.compile_json_schema(json.dumps(schema))

        # Initialize the combined processor
        # decoding_recorder = XGrammarDecodingRecorder(tokenizer, compiled_grammar, save_log=True)

        # Tokenize prompt into ids
        prompt = "Generate json object according to the following schema: \n"
        input_ids = tokenizer(
            [prompt], add_special_tokens=False, return_tensors="pt", padding=True
        )["input_ids"]
        input_ids = input_ids.to(model.device)

        # Inference Loop
        outputs = []
        # Generate sequences
        output = model.generate(
            input_ids,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            # logits_processor=[decoding_recorder],
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Detokenize generate output
        input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
        generated_tokens = output.sequences[:, input_length:]
        generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputs.append(generations[0])
        # Save the generated output to a JSON file
        output_entry = {
            "schema": schema_file.name,
            "output": generations[0]  # Since we only have one output per schema
        }
        
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            if idx > 0:
                f.write(',\n')
            f.write(json.dumps(output_entry, indent=4))

        # print("Generated outputs: ", outputs)
        # decoding_history = decoding_recorder.get_decoding_history()
        # entry = {
        #     "schema": schema_file.name,
        #     "output": outputs,
        #     "decoding_history": decoding_history,
        # }

        # Save the decoding history
        # print(f"Saving decoding history to {HISTORY_PATH}")
        # with open(HISTORY_PATH + schema_file.name, "w") as f:
        #     json.dump(entry, f, indent=4)

        idx += 1

if __name__ == "__main__":
    main() 