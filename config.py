import os
import torch

class Config(object):
    
    def __init__(self) -> None:

        self.DEVICE = torch.device("cpu:0")
        
        self.BATCH = 32
        self.EPOCHS = 5
        
        self.VOCAB_FILE = 'word2index5000.txt'
        self.VOCAB_SIZE = 5000
        
        self.NUM_LAYER = 1
        self.IMAGE_EMB_DIM = 512
        self.WORD_EMB_DIM = 512
        self.HIDDEN_DIM = 1024
        self.LR = 0.001
        
        self.EMBEDDING_WEIGHT_FILE = 'C:/PERSONAL/ImageCaption_Flickr30k-main/code/checkpoints/embeddings-32B-1024H-1L-e5.pt'
        self.ENCODER_WEIGHT_FILE = 'C:/PERSONAL/ImageCaption_Flickr30k-main/code/checkpoints/encoder-32B-1024H-1L-e5.pt'
        self.DECODER_WEIGHT_FILE = 'C:/PERSONAL/ImageCaption_Flickr30k-main/code/checkpoints/decoder-32B-1024H-1L-e5.pt'
        
        self.ROOT = os.path.join(r'C:/PERSONAL/ImageCaption_Flickr30k-main') 

        
        
