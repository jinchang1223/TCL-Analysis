import torch
import torch.nn.functional as F
import json
import logging

class TrieNode:
    def __init__(self, 
                 token_id=None, raw_likelihood=None, raw_score=None, 
                 success_rate=1, 
                 is_start_of_sequence=False, is_end_of_sequence=False,
                 eos_token_id=2):
        self.children = {}
        self.parent = None
        self.token_id = token_id
        self.raw_likelihood = raw_likelihood
        self.raw_score = raw_score
        
        # The default approximation of EFG
        self.success_rate = success_rate

        self.eos_token_id = eos_token_id
        self.is_start_of_sequence = is_start_of_sequence
        self.is_end_of_sequence = is_end_of_sequence

    def insert(self, child_node):
        """
        Insert child_node into the children dictionary        
        """
        if child_node.token_id not in self.children:
            self.children[child_node.token_id] = child_node
            child_node.parent = self 
            
            if child_node.token_id == self.eos_token_id:
                child_node.is_end_of_sequence = True

            # update the success rate of the parent node
            return self.update_success_rate()
        else:
            return 0

    def insert_accepted_tokens(self, scores, acceptance):
        """
        Create node from acceptance and scores and 
        insert as children of self node 
        """
        likelihoods = F.softmax(scores, dim=-1)

        for batch_index in range(acceptance.size(0)):
            accepted_tokens = acceptance[batch_index].nonzero().squeeze(-1)

            for token_id in accepted_tokens:
                if token_id not in self.children:
                    raw_likelihood = likelihoods[batch_index, token_id].item()
                    raw_score = scores[batch_index, token_id].item()

                    child_node = TrieNode(
                        token_id=token_id.item(),
                        raw_likelihood=raw_likelihood, 
                        raw_score=raw_score)

                    self.insert(child_node)

    def get_success_rate(self, token_id):
        """
        Return Approximated Expected Future Grammaticality of the token_id
        """
        if token_id in self.children:
            return self.children[token_id].success_rate
        else:
            return 1

    def update_success_rate(self):
        """
        Re-compute the success rate from the updated success rate of children
        """
        if self.children:
            total_success_rate = sum(child.raw_likelihood * child.success_rate for child in self.children.values())
            
            # Get how much of unexplored nodes are covered with this update
            updated_rate = self.success_rate - total_success_rate
            self.success_rate = total_success_rate

            # Back propagate the success rate
            if self.parent:
                return self.parent.update_success_rate()
            
            return updated_rate

    def prefix_raw_likelihood(self):
        if self.parent:
            return self.raw_likelihood * self.parent.prefix_raw_likelihood()
        else:
            return self.raw_likelihood

    def search_token(self, token_id):
        """
        Check if the self node has a children with token_id
        Return the children node if it exists, return None otherwise
        """
        if token_id in self.children:
            return self.children[token_id]
        else:
            return None

    def to_dict(self):
        """
        Convert a trie into a dictionary by removing the pointer to the parent
        """
        return {
            "token_id": self.token_id,
            "raw_likelihood": self.raw_likelihood,
            "raw_score": self.raw_score,
            "success_rate": self.success_rate,
            "eos_token_id": self.eos_token_id,
            "is_start_of_sequence": self.is_start_of_sequence,
            "is_end_of_sequence": self.is_end_of_sequence,
            "children": [child.to_dict() for child in self.children.values()]
        }

    @staticmethod
    def from_dict(d):
        """
        Recursively (re)construct trie from dictionary
        """
        node = TrieNode(
                 token_id=d['token_id'], 
                 raw_likelihood=d['raw_likelihood'], 
                 raw_score=d['raw_score'], 
                 success_rate=d['success_rate'], 
                 is_start_of_sequence=d['is_start_of_sequence'], 
                 is_end_of_sequence=d['is_end_of_sequence'],
                 eos_token_id=d['eos_token_id'])

        node.children = {child['token_id']:TrieNode.from_dict(child) for child in node.children}
        for child in node.children.values():
            child.parent = node

        return node

    def __repr__(self):
        parent_token_id = 'None (Root Node)' if self.parent is None else self.parent.token_id
        return (f"TrieNode(token_id={self.token_id}', "
                f"raw_likelihood={self.raw_likelihood}, raw_score={self.raw_score}, children={list(self.children.keys())}, "
                f"parent={parent_token_id}, success rate={self.success_rate})")

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.root.is_start_of_sequence = True

    def search_last_parent(self, prefix: torch.LongTensor):
        """
        Search the longest prefix in the trie that matches to the input sequence of tokens 'prefix'
        """
        matched_prefix = []
        current_parent = self.root

        # Assume one batch of prefix
        for time_step, token_id in enumerate(prefix[0]):
            token_id = token_id.item()
            if token_id in current_parent.children:
                current_parent = current_parent.children[token_id]
                matched_prefix.append(current_parent.token_id)
            else:
                print(
                    f"matched prefix is {matched_prefix}; current {token_id} not found in the trie at time step {time_step}")
                return None
        
        return current_parent

    def search(self, sequence):
        """
        Return the sequence of nodes that exactly matches with the input
        """
        node = self.root
        nodes = []
        for token_id in sequence:
            if token_id not in node.children:
                return None
            node = node.children[token_id]
            nodes.append(node)
        return nodes

    def raw_likelihood(self, sequence):
        """
        Return the raw likelihood (before the adjustment) of sequence
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()

        nodes = self.search(sequence)
        if nodes is None:
            return None

        likelihood = 1
        for node in nodes:
            likelihood *= node.raw_likelihood
        return likelihood

    def json(self):
        return json.dumps(self.root.to_dict(), indent=2)

    @staticmethod
    def loads(js):
        trie = Trie()
        trie.root = TrieNode.from_dict(json.loads(js))

        return trie

    def print_trie(self, node=None, prefix=None):
        """
        Print all the leaves in the trie
        """
        if node is None:
            node = self.root
        if prefix is None:
            prefix = []

        # If current node marks the end of a sequence, print the prefix as a list
        if node.is_end_of_sequence or len(node.children) == 0:
            print(prefix)

        # Recursively call print_trie for all children, appending the current character/token to the prefix
        for char, child_node in node.children.items():
            self.print_trie(child_node, prefix + [char])

    def has_full_information(self):
        """
        Checks if all paths in the trie end with an is_end_of_sequence node set to True.
        Returns True if the trie has full information, False otherwise.
        """
        return self._check_full_information(self.root)

    def _check_full_information(self, node):
        # If the node has no children, check if it is marked as the end of a sequence
        if not node.children:
            return node.is_end_of_sequence

        # Recursively check all children
        return all(self._check_full_information(child) for child in node.children.values())

    def print_all_nodes(self, node=None, depth=0):
        """
        Print all the nodes in the trie (including non-leaves)
        """

        if node is None:
            node = self.root

        # Print current node's details
        indent = "  " * depth  # Create indentation based on the depth in the trie
        node_details = (f"{indent}TrieNode(token_id={node.token_id}', "
                        f"raw_likelihood={node.raw_likelihood}, raw_score={node.raw_score}, success rate={node.success_rate}, "
                        f"children={list(node.children.keys())}, "
                        f"parent={node.parent.token_id if node.parent else None}, "
                        f"is_end_of_sequence={node.is_end_of_sequence})")
        print(node_details)

        # Recursively call print_all_nodes for all children
        for child_node in node.children.values():
            self.print_all_nodes(child_node, depth + 1)

    def get_tree_stats(self):
        """
        Get statistics about the trie tree structure
        
        Returns:
            dict: Dictionary containing tree statistics including:
                - total_nodes: Total number of nodes in the tree
                - max_depth: Maximum depth of the tree
                - root_children: Number of children of the root node
        """
        def count_nodes(node):
            """Count total number of nodes in the tree"""
            count = 1  # Count current node
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        def get_max_depth(node, current_depth=0):
            """Get the maximum depth of the tree"""
            if not node.children:
                return current_depth
            return max(get_max_depth(child, current_depth + 1) 
                      for child in node.children.values())
        
        return {
            "total_nodes": count_nodes(self.root),
            "max_depth": get_max_depth(self.root),
            "root_children": len(self.root.children)
        }