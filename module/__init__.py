from .tokenizer import Tokenizer
from .model import *
from .seq2seq import GRUEncoder, GRUDecoder, Attention, Seq2SeqModel, build_seq2seq_model
import argparse


def init_model_by_key(args, tokenizer: Tokenizer):
    key = args.model.lower()
    if key == 'transformer':
        model = Transformer(tokenizer.vocab_size, args.max_seq_len, args.embed_dim, args.hidden_dim, args.n_layer, args.n_head, args.ff_dim, args.embed_drop, args.hidden_drop)
    elif key == 'bilstm':
        model = BiLSTM(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop)
    elif key == 'cnn':
        model = CNN(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.embed_drop)
    elif key == 'bilstmattn':
        model = BiLSTMAttn(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop, args.n_head)
    elif key == 'bilstmcnn':
        model = BiLSTMCNN(tokenizer.vocab_size, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop)
    elif key == 'bilstmconvattres':
        model = BiLSTMConvAttRes(tokenizer.vocab_size, args.max_seq_len, args.embed_dim, args.hidden_dim, args.n_layer, args.embed_drop, args.hidden_drop, args.n_head)
    elif key == 'seq2seq':
        # Seq2Seq with Attention
        model = build_seq2seq_model(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            n_layer=args.n_layer,
            dropout=args.embed_drop,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            use_attention=True,
            bidirectional=True
        )
    elif key == 'seq2seq_noattn':
        # Seq2Seq without Attention
        model = build_seq2seq_model(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            n_layer=args.n_layer,
            dropout=args.embed_drop,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            use_attention=False,
            bidirectional=True
        )
    else:
        raise KeyError(f"Model `{args.model}` does not exist")
    return model


def is_seq2seq_model(model):
    """判断模型是否为Seq2Seq模型"""
    return isinstance(model, Seq2SeqModel)