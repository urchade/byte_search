import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class ByteCharEncoder(object):
    
    def __init__(self):
        
        vocab = None
                
    def encode_token(self, token):
        """encode a single token: str"""
        return torch.LongTensor(list(token.encode("utf-8"))) + 1 # 0 for padding
    
    def encode_sequence(self, sequence):
        """encode a sequence of token: [str] or str"""
        return [self.encode_token(token) for token in sequence]
    
    def encode_batch(self, batch):
        """encode a batch: [[str]] or [str]"""
        
        lengths = []
        batch_tokens = []
        
        for seq in batch:
            batch_tokens.extend(self.encode_sequence(seq))
            lengths.append(len(seq))
            
        batch_tokens = pad_sequence(batch_tokens, batch_first=True)
            
        list_seq = torch.split(batch_tokens, lengths)
            
        return pad_sequence(list_seq, batch_first=True)
    
    def encode_batch_with_length(self, batch, length=15):
        
        lengths = []
        batch_tokens = []
        
        for seq in batch:
            batch_tokens.extend(self.encode_sequence(seq))
            lengths.append(len(seq))
        
        batch_tokens = torch.stack([self.pad_middle(i, length) for i in batch_tokens], dim=0)
        
        list_seq = torch.split(batch_tokens, lengths)
        
        return pad_sequence(list_seq, batch_first=True)
    
    def pad_middle(self, b, max_len):
        
        b = b[:max_len]
        
        len_b = len(b)
        
        pad_left = (max_len - len_b)//2
        
        if (len_b + max_len) % 2 == 0:
            return F.pad(b, (pad_left, pad_left))
        else:
            return F.pad(b, (pad_left + 1, pad_left))