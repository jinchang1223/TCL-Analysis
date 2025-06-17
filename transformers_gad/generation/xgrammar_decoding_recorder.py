import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor
import xgrammar
from transformers_gad.oracle.oracle_trie import Trie

class XGrammarDecodingRecorder(LogitsProcessor):
    """
    A logits processor that combines xgrammar's grammar constraint with decoding tree recording.
    This class uses xgrammar for grammar-constrained generation while recording the complete
    decoding tree structure.
    """
    def __init__(self, tokenizer, compiled_grammar, oracle_trie=Trie(), save_log=False):
        self.tokenizer = tokenizer
        
        # XGrammar processor
        self.xgr_processor = xgrammar.contrib.hf.LogitsProcessor(compiled_grammar)
        
        # Oracle trie for recording decoding tree
        self.oracle_trie = oracle_trie
        
        # Generation Log
        self.save_log = save_log
        self.history = []
        
        # Store the compiled grammar for token decoding
        self.compiled_grammar = compiled_grammar
        # Store the tokenizer info
        self.tokenizer_info = compiled_grammar.tokenizer_info
        
        # Store previous step's scores for updating probabilities
        self.prev_scores = None
        self.prev_likelihoods = None
        
        # Step counter
        self.current_step = 0

    def record_decoding_tree(self, scores, acceptance, selected_token=None):
        """
        Record the decoding tree by storing accepted tokens in the trie
        
        Args:
            scores (torch.FloatTensor): The logits from the language model
            acceptance (torch.Tensor): Boolean tensor indicating accepted tokens
            selected_token (int, optional): The token ID that was actually selected in this step
        """
        # # Find the current parent node in the trie
        # current_parent = self.oracle_trie.search_last_parent(self.generated_tokens)
        
        # # If no parent node is found, use the root node
        # if current_parent is None:
        #     current_parent = self.oracle_trie.root
        #     print(f"Warning: No parent node found in trie, using root node instead.")
        
        # # Record accepted tokens in the trie
        # current_parent.insert_accepted_tokens(scores, acceptance)

        if self.save_log:
            self.store_decoding_history(acceptance, scores, selected_token)

    def process_scores(self, input_ids, scores, next_tokens=None):
        """
        Process the scores using xgrammar and record the decoding tree
        
        Args:
            input_ids (torch.LongTensor): The input token IDs
            scores (torch.FloatTensor): The logits from the language model
            next_tokens (torch.LongTensor, optional): The actual tokens selected by the model
            
        Returns:
            torch.FloatTensor: The scores after applying xgrammar constraints
        """
        # Store generated tokens for tree recording
        self.generated_tokens = input_ids

        # Apply xgrammar constraints
        masked_scores = self.xgr_processor(input_ids, scores)
        
        # Get acceptance mask from xgrammar processor
        acceptance = (masked_scores != float('-inf'))
        
        # Print debugging information
        # Allowed token IDs for the current step
        allowed_token_ids = torch.nonzero(acceptance[0]).squeeze(-1).tolist()
        print("\n===[DEBUG] Allowed token ids this step ===")
        print(allowed_token_ids)
        print("===[DEBUG] Allowed token strings ===")
        print([self.tokenizer.decode([tid]) for tid in allowed_token_ids])
        
        # Get the actual selected token from next_tokens if provided
        selected_token = next_tokens[0].item() if next_tokens is not None else None
        
        # Record the decoding tree with masked scores
        self.record_decoding_tree(scores, acceptance, selected_token)
        
        return masked_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process the scores using xgrammar and record the decoding tree.
        This method is called by the transformers generation process.
        
        Args:
            input_ids (torch.LongTensor): The input token IDs, where the last token is the one selected in the previous step
            scores (torch.FloatTensor): The logits from the language model for the current step
            
        Returns:
            torch.FloatTensor: The scores after applying xgrammar constraints
        """
        # Get the token that was selected in the previous step
        # The last token in input_ids is the one that was just selected in the previous step
        if input_ids.size(1) > 0:
            prev_selected_token = input_ids[0, -1].item()
        else:
            prev_selected_token = None
            
        # Process scores and record decoding tree with the previous step's selected token
        return self.process_scores(input_ids, scores, torch.tensor([[prev_selected_token]]) if prev_selected_token is not None else None)

    def reset(self):
        """Reset the processor state"""
        self.reset_history()
        self.reset_trie()
        self.prev_scores = None
        self.prev_likelihoods = None
        self.current_step = 0  # Reset step counter

    def reset_history(self):
        """Reset the history log"""
        self.history = []

    def reset_trie(self):
        """Reset the oracle trie"""
        self.oracle_trie = Trie()

    def store_decoding_history(self, acceptance, scores, selected_token=None):
        """
        Store information about accepted tokens in the current decoding step
        
        Args:
            acceptance (torch.Tensor): Boolean tensor indicating accepted tokens
            scores (torch.Tensor): The logits from the language model
            selected_token (int, optional): The token ID that was selected in the previous step
        """
        likelihoods = F.softmax(scores, dim=-1)

        batch_accepted_info = []

        for batch_index in range(acceptance.size(0)):
            accepted_info = []
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)
            
            # Calculate metrics
            n_acc = len(accepted_indices)  # Number of accepted tokens
            n_vocab = acceptance.size(-1)  # Total vocabulary size
            acceptance_ratio = n_acc / n_vocab  # Acceptance ratio
            
            # Calculate total probability mass of accepted tokens
            accepted_probs = likelihoods[batch_index, accepted_indices]
            total_accepted_prob = likelihoods[batch_index, accepted_indices].sum().item()
            total_prob_before_filter = likelihoods[batch_index].sum().item()
            prob_mass = {
                "total_prob_before_filter": total_prob_before_filter,
                "total_prob_after_filter": total_accepted_prob,
                "max_prob_before_filter": likelihoods[batch_index].max().item(),
                "min_prob_before_filter": likelihoods[batch_index].min().item(),
                "max_prob_after_filter": accepted_probs.max().item(),
                "min_prob_after_filter": accepted_probs.min().item(),
            }
            
            # Get top token information for current step
            top_token_id = scores[batch_index].argmax().item()
            top_token_prob = likelihoods[batch_index, top_token_id].item()
            top_token = self.tokenizer.convert_ids_to_tokens(top_token_id)
            is_intervened = not acceptance[batch_index, top_token_id].item()
            
            # Get Top-20 tokens before constraint
            top_20_before = []
            top_20_indices_before = torch.topk(likelihoods[batch_index], 20)
            for idx, prob in zip(top_20_indices_before.indices, top_20_indices_before.values):
                token_id = idx.item()
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                top_20_before.append({
                    "token_id": token_id,
                    "token": token,
                    "raw_likelihood": prob.item()
                })
            
            # Get Top-20 tokens after constraint
            top_20_after = []
            if len(accepted_indices) > 0:
                top_20_indices_after = torch.topk(accepted_probs, min(20, len(accepted_indices)))
                for idx, prob in zip(top_20_indices_after.indices, top_20_indices_after.values):
                    token_id = accepted_indices[idx].item()
                    token = self.tokenizer.convert_ids_to_tokens(token_id)
                    top_20_after.append({
                        "token_id": token_id,
                        "token": token,
                        "raw_likelihood": prob.item()
                    })
            
            # Initialize previous step's token probability as None
            actual_token_prob = None
            actual_token = None
            
            # If we have the previous step's selected token, calculate its probability
            if selected_token is not None:
                # Update previous step's probability if we have it
                if self.prev_likelihoods is not None and len(self.history) > 0:
                    prev_step = self.history[-1][batch_index]
                    if "metrics" in prev_step:
                        prev_step["metrics"]["actual_token_prob"] = self.prev_likelihoods[batch_index, selected_token].item()
                        prev_step["metrics"]["actual_token"] = self.tokenizer.convert_ids_to_tokens(selected_token)

            for idx in accepted_indices:
                token_id = idx.item()
                raw_score = scores[batch_index, idx].item()
                likelihood = likelihoods[batch_index, idx].item()
                # Convert token ID to string representation
                token = self.tokenizer.convert_ids_to_tokens(token_id)

                accepted_info.append({
                    "token_id": token_id,
                    "token": token,
                    "raw_score": raw_score,
                    "raw_likelihood": likelihood,
                })

            # Add metrics to the batch info
            batch_info = {
                # "accepted_tokens": accepted_info,
                "metrics": {
                    "step": self.current_step,  # Add current step number
                    "n_accepted": n_acc,
                    "acceptance_ratio": acceptance_ratio,
                    "prob_mass": prob_mass,
                    "top_token_prob": top_token_prob,
                    "top_token": top_token,
                    "actual_token_prob": actual_token_prob,  # Probability of the token selected in the previous step
                    "actual_token": actual_token,
                    "is_intervened": is_intervened,
                },
                "top_20_before_constraint": top_20_before,
                "top_20_after_constraint": top_20_after
            }
            
            batch_accepted_info.append(batch_info)

        self.history.append(batch_accepted_info)
        
        # Store current scores and likelihoods for next step
        self.prev_scores = scores
        self.prev_likelihoods = likelihoods
        
        # Increment step counter
        self.current_step += 1

    def get_decoding_tree(self):
        """
        Get the recorded decoding tree
        
        Returns:
            Trie: The oracle trie containing the complete decoding tree
        """
        return self.oracle_trie

    def get_decoding_history(self):
        """
        Get the recorded decoding history
        
        Returns:
            list: List of dictionaries containing information about accepted tokens at each step
        """
        return self.history 