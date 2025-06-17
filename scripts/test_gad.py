import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor

NUM_ITER = 10
MODEL_ID = "TinyLlama/TinyLlama_v1.1"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
TRIE_PATH = "tries/binary_len_5_0_trie.json"
DEVICE = "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

device = torch.device(DEVICE)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)
model.to(dtype=DTYPE)
model.resize_token_embeddings(len(tokenizer))

# Load EBNF grammar
with open(GRAMMAR_PATH, "r") as file:
    grammar_str = file.read()
grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

# Initialize logits processor for the grammar
gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar)
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([
    inf_nan_remove_processor,
    gad_oracle_processor,
])

# Tokenize prompt into ids
prompt = "Generate a binary string of length 5"
input_ids = tokenizer(
    [prompt], add_special_tokens=False, return_tensors="pt", padding=True
)["input_ids"]
input_ids = input_ids.to(model.device)

# Inference Loop
outputs = []
for _ in tqdm(range(NUM_ITER), desc="Running Inference"):
    # Generate sequences
    output = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        logits_processor=logits_processors,
        repetition_penalty=REPETITION_PENALTY,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Incremental parser state must be reset after each generation
    gad_oracle_processor.reset()

    # Detokenize generate output
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs.append(generations[0])

print(outputs)

# Store the trie as JSON
import os
if not os.path.isdir("tries"):
    os.mkdir("tries")

with open(TRIE_PATH, "w") as f:
    f.write(gad_oracle_processor.oracle_trie.json())