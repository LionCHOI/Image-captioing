from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerDecoderLayer, LayerNorm, TransformerDecoder
from torch.nn import functional as F

import math
from typing import Optional, Any, Union, Callable

# global variable --------------------------------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # cuda 설정


# FUNCTIONS FOR MASKING ============================================================================================
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)                   # caption의 길이만큼 값이 True인 lower triangle matrix를 만든다.
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))    # False면 -inf, True면 0으로 대입한다.
    return mask


def create_mask(tgt_input, PAD_IDX):
    tgt_seq_len = tgt_input.shape[1]                            # caption의 길이를 받아온다.
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)     # caption의 masking을 받는다.
    tgt_padding_mask = (tgt_input == PAD_IDX)                   # caption의 padding 위치를 받는다. (padding masking에 활용하고자)
    
    return tgt_mask, tgt_padding_mask


# DECODER ===========================================================================================================
# 1) 위치 인코딩을 위한 모듈  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class PositionalEncoding(nn.Module):    
    
    # init ----------------------------------------------------------------------------------------------------------
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # positional embedding을 cos과 sin을 이용해서 생성
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        # dropout 설정
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    # forward -------------------------------------------------------------------------------------------------------
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])  # positional embedding 값은 연결시킨후 drop out 적용


# 2) 토큰 임베딩의 텐서로 변환하기 위한 모듈 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class TokenEmbedding(nn.Module):
    
    # init ----------------------------------------------------------------------------------------------------------
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size) # embedding 설정
        self.emb_size = emb_size                            # embedding size 저장

    # forward -------------------------------------------------------------------------------------------------------
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size) # 입력을 long 형식을 변환 후 embedding 후에 scale 값 적용


# 3) Decoder 신경망 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Seq2SeqDecoder(nn.Module):
    
    # init -----------------------------------------------------------------------------------------------------------
    def __init__(self, d_model: int = 512, nhead: int = 8, tgt_vocab_size: int = 4468,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, norm_first: bool = False, device=None, dtype=None):
        """ Decoder 신경망 init

        Args:
            d_model (int, optional):                                        입출력 크기.                Defaults to 512.
            nhead (int, optional):                                          헤드의 개수.                Defaults to 8.
            tgt_vocab_size (int, optional):                                 vocabulary의 크기.          Defaults to 4468.
            num_decoder_layers (int, optional):                             decoder layer의 층 수.      Defaults to 6.
            dim_feedforward (int, optional):                                feedforward의 수.           Defaults to 2048.
            dropout (float, optional):                                      dropout 비율.               Defaults to 0.1.
            activation (Union[str, Callable[[Tensor], Tensor]], optional):  활성화 함수.                Defaults to F.relu.
            layer_norm_eps (float, optional):                               layer norm의 엡실론 값.     Defaults to 1e-5.
            batch_first (bool, optional):                                   batch first의 불리언 값.    Defaults to False.
            norm_first (bool, optional):                                    norm first의 불리언 값.     Defaults to False.
            device (_type_, optional):                                      device 종류.                Defaults to None.
            dtype (_type_, optional):                                       type 종류.                  Defaults to None.
        """
        super(Seq2SeqDecoder, self).__init__()
        
        # 변수 저장 ------------------------------------------------------------------------------------------------------
        factory_kwargs = {'device': device, 'dtype': dtype} # factory_kwargs로 device와 dtype 저장
        
        self.device = device        # device 저장
        self.emb_size = d_model     # 입출력 크기 저장
        
        # module 설정 ----------------------------------------------------------------------------------------------------
        self.positional_encoding = PositionalEncoding(self.emb_size, dropout=dropout)   # positional embedding
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.emb_size)                # token embedding
        
        # decoder = decoder layer + decoder norm 
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, 
                                                dropout, activation, layer_norm_eps, 
                                                batch_first, norm_first, **factory_kwargs)  # decoder layer
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)             # decoder norm 
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)  # decoder
        
        self.generator = nn.Linear(self.emb_size, tgt_vocab_size)   # FC layer
        self.dropout = nn.Dropout(dropout)                          # drop out layer
        
    # forward -------------------------------------------------------------------------------------------------------
    def forward(self, tgt_input: Tensor, img_features: Tensor, tgt_mask: Tensor, tgt_padding_mask: Optional[Tensor] = None):
        """ forward

        Args:
            tgt_input (Tensor): caption
            img_features (Tensor): encoder output
            tgt_mask (Tensor): caption mask
            tgt_padding_mask (Optional[Tensor], optional): caption padding mask. Defaults to None.

        Returns:
            decoder output after FC layer
        """
        tgt = self.positional_encoding(self.tgt_tok_emb(tgt_input))                                                 # token embedding + position embedding with caption 
        return self.generator(self.decoder(tgt, img_features, tgt_mask, tgt_key_padding_mask=tgt_padding_mask))     # FC layer + decoder with cpation token

    # decode --------------------------------------------------------------------------------------------------------
    def decode(self, tgt: Tensor, img_features: Tensor, tgt_mask: Tensor):
        """ decode for evaluation

        Args:
            tgt (Tensor): new caption starting with <SOS>
            img_features (Tensor): encoder output
            tgt_mask (Tensor): new caption mask 

        Returns:
            tensor: decoder output with new caption
        """
        return self.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), img_features, tgt_mask)  # token embedding + position embedding + decoder with cpation


# FUNCTIONS FOR EVALUATION =========================================================================================
def translate(model: torch.nn.Module, img_feature: Tensor, max_len :int, start_symbol: int, end_symbol: int):
    
    # 변수 선언 -----------------------------------------------------------------------------------------------------
    batch_size = img_feature.shape[0]                                               # batch size를 저장
    pred_sentence = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(DEVICE)  # decoder의 입력으로 들어갈 <SOS> 선언
    
    # decoder 수행 --------------------------------------------------------------------------------------------------
    for i in range(max_len-1):          # 반복문 수행
        tgt_mask = (generate_square_subsequent_mask(pred_sentence.size(1)).type(torch.bool)).to(DEVICE)             # 입력과 함께 들어갈 masking 생성
        out = model.decode(pred_sentence, img_feature, tgt_mask)                                                    # decoder 수행
        prob = model.generator(out[:, -1])                                                                          # decoder의 값을 FC 수행 
        _, next_word = torch.max(prob, dim=1)                                                                       # FC의 값에서 최종 단어 선택
        pred_sentence = torch.cat([pred_sentence, next_word.reshape(batch_size, -1).type_as(pred_sentence)], dim=1) # 최종 단어를 입력과 연결
        
        if next_word == end_symbol:     # 만약 최종 단어가 end symbol이면  
            break                       # 반복문 종료
        
    return pred_sentence