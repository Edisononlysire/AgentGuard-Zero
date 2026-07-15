from __future__ import annotations

import torch
from transformers import StoppingCriteria


class CompleteJSONObjectCriteria(StoppingCriteria):
    """Stop each sequence after its first complete top-level JSON object."""

    def __init__(self, tokenizer, batch_size: int):
        self.tokenizer = tokenizer
        self.started = [False] * batch_size
        self.depth = [0] * batch_size
        self.in_string = [False] * batch_size
        self.escaped = [False] * batch_size
        self.done = [False] * batch_size
        self._piece_cache: dict[int, str] = {}

    def __call__(self, input_ids, scores, **kwargs):
        token_ids = input_ids[:, -1].detach().cpu().tolist()
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            piece = self._piece_cache.get(token_id)
            if piece is None:
                piece = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                self._piece_cache[token_id] = piece
            pieces.append(piece)
        for index, piece in enumerate(pieces):
            if self.done[index]:
                continue
            for char in piece:
                if not self.started[index]:
                    if char == "{":
                        self.started[index] = True
                        self.depth[index] = 1
                    continue
                if self.in_string[index]:
                    if self.escaped[index]:
                        self.escaped[index] = False
                    elif char == "\\":
                        self.escaped[index] = True
                    elif char == '"':
                        self.in_string[index] = False
                    continue
                if char == '"':
                    self.in_string[index] = True
                elif char == "{":
                    self.depth[index] += 1
                elif char == "}":
                    self.depth[index] -= 1
                    if self.depth[index] == 0:
                        self.done[index] = True
                        break
        return torch.tensor(self.done, dtype=torch.bool, device=input_ids.device)
