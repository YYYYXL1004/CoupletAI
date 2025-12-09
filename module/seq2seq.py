"""
Seq2Seq模型实现
包含: GRUEncoder, Attention, GRUDecoder, Seq2SeqModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GRUEncoder(nn.Module):
    """
    GRU编码器
    - 支持双向GRU
    - 支持多层堆叠
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_layer: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
    
    def forward(self, src):
        """
        Args:
            src: (batch_size, src_len)
        Returns:
            outputs: (batch_size, src_len, hidden_dim * n_directions)
            hidden: (n_layer, batch_size, hidden_dim * n_directions)
        """
        # (batch_size, src_len, embed_dim)
        embedded = self.dropout(self.embedding(src))
        
        # outputs: (batch_size, src_len, hidden_dim * n_directions)
        # hidden: (n_layer * n_directions, batch_size, hidden_dim)
        outputs, hidden = self.gru(embedded)
        
        # 如果是双向的，需要合并前向和后向的hidden state
        if self.bidirectional:
            # hidden: (n_layer, 2, batch_size, hidden_dim) -> (n_layer, batch_size, hidden_dim * 2)
            hidden = hidden.view(self.n_layer, 2, -1, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1)
        
        return outputs, hidden


class Attention(nn.Module):
    """
    Bahdanau Attention (加性注意力)
    """
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: (batch_size, decoder_dim) - 解码器当前隐藏状态
            encoder_outputs: (batch_size, src_len, encoder_dim)
            mask: (batch_size, src_len) - padding mask
        Returns:
            attention_weights: (batch_size, src_len)
            context: (batch_size, encoder_dim)
        """
        src_len = encoder_outputs.shape[1]
        
        # hidden: (batch_size, decoder_dim) -> (batch_size, src_len, decoder_dim)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # (batch_size, src_len, encoder_dim + decoder_dim)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=-1)))
        
        # (batch_size, src_len)
        attention = self.v(energy).squeeze(-1)
        
        if mask is not None:
            attention = attention.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(attention, dim=-1)
        
        # (batch_size, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return attention_weights, context


class GRUDecoder(nn.Module):
    """
    带Attention的GRU解码器
    """
    def __init__(self, vocab_size: int, embed_dim: int, encoder_dim: int,
                 decoder_dim: int, n_layer: int = 1, dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.n_layer = n_layer
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if use_attention:
            self.attention = Attention(encoder_dim, decoder_dim)
            gru_input_dim = embed_dim + encoder_dim
        else:
            gru_input_dim = embed_dim
        
        self.gru = nn.GRU(
            gru_input_dim,
            decoder_dim,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0,
            batch_first=True
        )
        
        if use_attention:
            self.fc_out = nn.Linear(decoder_dim + encoder_dim + embed_dim, vocab_size)
        else:
            self.fc_out = nn.Linear(decoder_dim, vocab_size)
    
    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        单步解码
        Args:
            input_token: (batch_size,) - 当前输入token
            hidden: (n_layer, batch_size, decoder_dim)
            encoder_outputs: (batch_size, src_len, encoder_dim)
            mask: (batch_size, src_len)
        Returns:
            output: (batch_size, vocab_size)
            hidden: (n_layer, batch_size, decoder_dim)
            attention_weights: (batch_size, src_len) or None
        """
        # (batch_size, 1, embed_dim)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        
        if self.use_attention:
            # 使用最后一层的hidden state计算attention
            attn_weights, context = self.attention(hidden[-1], encoder_outputs, mask)
            
            # (batch_size, 1, embed_dim + encoder_dim)
            gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            gru_input = embedded
            attn_weights = None
            context = None
        
        # output: (batch_size, 1, decoder_dim)
        # hidden: (n_layer, batch_size, decoder_dim)
        output, hidden = self.gru(gru_input, hidden)
        
        # (batch_size, decoder_dim)
        output = output.squeeze(1)
        
        if self.use_attention:
            # (batch_size, decoder_dim + encoder_dim + embed_dim)
            output = torch.cat([output, context, embedded.squeeze(1)], dim=-1)
        
        # (batch_size, vocab_size)
        prediction = self.fc_out(output)
        
        return prediction, hidden, attn_weights


class Seq2SeqModel(nn.Module):
    """
    完整的Seq2Seq模型
    """
    def __init__(self, encoder: GRUEncoder, decoder: GRUDecoder, 
                 sos_id: int, eos_id: int, pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        # 如果encoder是双向的，需要投影到decoder的hidden维度
        encoder_hidden_dim = encoder.hidden_dim * encoder.n_directions
        decoder_hidden_dim = decoder.decoder_dim
        
        if encoder_hidden_dim != decoder_hidden_dim:
            self.hidden_proj = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        else:
            self.hidden_proj = None
    
    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len) - 上联
            trg: (batch_size, trg_len) - 下联 (训练时提供)
            teacher_forcing_ratio: 使用teacher forcing的概率
        Returns:
            outputs: (batch_size, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        
        if trg is not None:
            trg_len = trg.shape[1]
        else:
            # 推理时，生成与输入等长的序列
            trg_len = src.shape[1]
        
        # 存储每一步的输出
        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(src.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src)
        
        # 投影hidden到decoder维度
        if self.hidden_proj is not None:
            hidden = torch.tanh(self.hidden_proj(hidden))
        
        # 创建mask
        mask = (src == self.pad_id)
        
        # 第一个输入是SOS
        input_token = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=src.device)
        
        for t in range(trg_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t, :] = output
            
            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio if trg is not None else False
            
            if teacher_force:
                input_token = trg[:, t]
            else:
                input_token = output.argmax(dim=-1)
        
        return outputs
    
    def generate(self, src, max_len=None, repetition_penalty=1.2, 
                 punctuation_ids=None, force_same_length=True):
        """
        推理时的生成方法（带重复惩罚的贪婪解码）
        Args:
            src: (batch_size, src_len)
            max_len: 最大生成长度，默认等于src长度
            repetition_penalty: 重复惩罚系数，>1会抑制重复
            punctuation_ids: 标点符号token id列表，用于位置对齐
            force_same_length: 强制生成与输入等长的序列
        """
        self.eval()
        batch_size = src.shape[0]
        
        # 计算每个样本的实际长度（非PAD部分）
        if force_same_length:
            src_lens = (src != self.pad_id).sum(dim=1)  # (batch_size,)
            trg_len = src.shape[1]
        else:
            trg_len = max_len if max_len else src.shape[1]
            src_lens = torch.full((batch_size,), trg_len, device=src.device)
        
        device = src.device
        punct_set = set(punctuation_ids) if punctuation_ids else set()
        
        with torch.no_grad():
            # 编码
            encoder_outputs, hidden = self.encoder(src)
            
            if self.hidden_proj is not None:
                hidden = torch.tanh(self.hidden_proj(hidden))
            
            mask = (src == self.pad_id)
            
            outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(device)
            input_token = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=device)
            
            # 记录已生成的token用于重复惩罚
            generated = [[] for _ in range(batch_size)]
            
            for t in range(trg_len):
                output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in generated[i]:
                            output[i, token_id] /= repetition_penalty
                
                # 标点符号位置对齐：上联是标点则下联也必须是标点，反之禁止标点
                if punct_set:
                    for i in range(batch_size):
                        if t < src_lens[i]:
                            src_token = src[i, t].item()
                            if src_token in punct_set:
                                # 上联此位置是标点，强制输出相同标点
                                output[i, :] = float('-inf')
                                output[i, src_token] = 0
                            else:
                                # 上联此位置不是标点，禁止输出标点
                                for punct_id in punct_set:
                                    output[i, punct_id] = float('-inf')
                
                # 对于已经达到目标长度的样本，强制输出PAD
                if force_same_length:
                    for i in range(batch_size):
                        if t >= src_lens[i]:
                            output[i, :] = float('-inf')
                            output[i, self.pad_id] = 0
                
                outputs[:, t, :] = output
                input_token = output.argmax(dim=-1)
                
                # 记录生成的token
                for i in range(batch_size):
                    generated[i].append(input_token[i].item())
            
            return outputs


def build_seq2seq_model(vocab_size: int, embed_dim: int, hidden_dim: int,
                        n_layer: int, dropout: float, sos_id: int, 
                        eos_id: int, pad_id: int, use_attention: bool = True,
                        bidirectional: bool = True):
    """
    构建Seq2Seq模型的工厂函数
    """
    encoder = GRUEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layer=n_layer,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    encoder_dim = hidden_dim * (2 if bidirectional else 1)
    decoder_dim = encoder_dim  # 保持一致
    
    decoder = GRUDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        n_layer=n_layer,
        dropout=dropout,
        use_attention=use_attention
    )
    
    model = Seq2SeqModel(encoder, decoder, sos_id, eos_id, pad_id)
    
    return model
