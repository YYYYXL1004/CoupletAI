import argparse
import logging
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# TensorBoard可选
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from module.model import BiLSTM, Transformer, CNN, BiLSTMAttn, BiLSTMCNN, BiLSTMConvAttRes
from module import Tokenizer, init_model_by_key, is_seq2seq_model
from module.metric import calc_bleu, calc_rouge_l


def setup_logger(log_file: Path):
    """设置日志：文件记录详细日志，终端只显示关键信息"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件handler - 详细日志
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=768, type=int)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--device", default="cuda:4", type=str)
    parser.add_argument("-m", "--model", default='transformer', type=str)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", default='O1', type=str)
    parser.add_argument("--max_grad_norm", default=3.0, type=float)
    parser.add_argument("--dir", default='dataset', type=str)
    parser.add_argument("--output", default='output', type=str)
    parser.add_argument("--logdir", default='runs', type=str)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--ff_dim", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)

    parser.add_argument("--test_epoch", default=1, type=int)

    parser.add_argument("--embed_drop", default=0.2, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    
    # Seq2Seq专用参数
    parser.add_argument("--teacher_forcing", default=0.5, type=float, help="Teacher forcing ratio for Seq2Seq")
    
    # 实验管理
    parser.add_argument("--exp_name", default=None, type=str, help="实验名称，用于区分不同实验")
    parser.add_argument("--patience", default=5, type=int, help="早停耐心值，连续多少个epoch没有提升就停止")
    return parser.parse_args()

def auto_evaluate(model, testloader, tokenizer, use_seq2seq=False):
    """评估模型，支持原始模型和Seq2Seq模型"""
    bleus = []
    rls = []
    device = next(model.parameters()).device
    model.eval()
    for batch in testloader:
        input_ids, masks, lens, target_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if use_seq2seq:
                # Seq2Seq模型：使用带重复惩罚的generate方法（评估时不做标点对齐，保持原始输出）
                logits = model.generate(input_ids, repetition_penalty=1.2, punctuation_ids=None)
            else:
                # 原始模型：序列标注
                logits = model(input_ids, masks)
            _, preds = torch.max(logits, dim=-1)
        
        for seq, tag in zip(preds.tolist(), target_ids.tolist()):
            seq = list(filter(lambda x: x != tokenizer.pad_id, seq))
            tag = list(filter(lambda x: x != tokenizer.pad_id, tag))
            bleu = calc_bleu(seq, tag)
            rl = calc_rouge_l(seq, tag)
            bleus.append(bleu)
            rls.append(rl)
    return sum(bleus) / len(bleus), sum(rls) / len(rls)


def calc_loss(model, dataloader, loss_function, device, use_seq2seq=False):
    """计算验证集loss"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            
            if use_seq2seq:
                logits = model(input_ids, trg=target_ids, teacher_forcing_ratio=0)
            else:
                logits = model(input_ids, masks)
            
            loss = loss_function(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    return total_loss / total_samples

def predict_demos(model, tokenizer: Tokenizer, use_seq2seq=False, logger=None):
    """预测demo样例"""
    demos = [
        "马齿草焉无马齿", "天古天今，地中地外，古今中外存天地", 
        "笑取琴书温旧梦", "日里千人拱手划船，齐歌狂吼川江号子",
        "我有诗情堪纵酒", "我以真诚溶冷血",
        "三世业岐黄，妙手回春人共赞"
    ]
    sents = [torch.tensor(tokenizer.encode(sent)).unsqueeze(0) for sent in demos]
    model.eval()
    device = next(model.parameters()).device
    results = []
    for i, sent in enumerate(sents):
        sent = sent.to(device)
        with torch.no_grad():
            if use_seq2seq:
                # Seq2Seq模型：使用带标点对齐的generate方法
                punct_ids = tokenizer.get_punctuation_ids()
                logits = model.generate(sent, repetition_penalty=1.5, punctuation_ids=punct_ids).squeeze(0)
            else:
                logits = model(sent).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = tokenizer.decode(pred)
        results.append((demos[i], pred))
        if logger:
            logger.info(f"上联：{demos[i]}。 预测的下联：{pred}")
    return results

def save_model(filename, model, args, tokenizer):
    info_dict = {
        'model': model.state_dict(),
        'args': args,
        'tokenzier': tokenizer
    }
    torch.save(info_dict, filename)

def run():
    args = get_args()
    fdir = Path(args.dir)
    
    # 设置实验名称
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{args.model}_L{args.n_layer}_H{args.hidden_dim}"
    
    # 设置输出目录和日志
    output_dir = Path(args.output) / exp_name
    output_dir.mkdir(exist_ok=True, parents=True)
    log_file = output_dir / "train.log"
    logger = setup_logger(log_file)
    
    # TensorBoard
    tb = SummaryWriter(Path(args.logdir) / exp_name)
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Args: {args}")
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 加载数据
    logger.info("Loading vocab and datasets...")
    tokenizer = Tokenizer.from_pretrained(fdir / 'vocab.pkl')
    train_dataset = torch.load(fdir / 'train.pkl')
    val_dataset = torch.load(fdir / 'val.pkl')
    test_dataset = torch.load(fdir / 'test.pkl')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 初始化模型
    logger.info("Initializing model...")
    model = init_model_by_key(args, tokenizer)
    model.to(device)
    use_seq2seq = is_seq2seq_model(model)
    
    logger.info(f"Model: {model.__class__.__name__}, Seq2Seq: {use_seq2seq}")
    print(f"Model: {model.__class__.__name__}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # FP16训练
    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, 'einsum')
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    
    # 数据并行
    use_dataparallel = False
    if torch.cuda.device_count() > 1 and device.type != 'cpu' and (not args.device or ':' not in args.device):
        model = torch.nn.DataParallel(model)
        use_dataparallel = True
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # 训练记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'bleu': [],
        'rouge_l': [],
        'epoch_time': []
    }
    
    global_step = 0
    best_val_loss = float('inf')
    no_improve_count = 0  # 早停计数器
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        t1 = time.time()
        epoch_loss = 0.0
        
        # 使用tqdm显示进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                    leave=True, ncols=100)
        
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            
            # 根据模型类型选择不同的前向传播
            if use_seq2seq:
                logits = model(input_ids, trg=target_ids, teacher_forcing_ratio=args.teacher_forcing)
            else:
                logits = model(input_ids, masks)
            
            loss = loss_function(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            
            if use_dataparallel:
                loss = loss.mean()
            
            epoch_loss += loss.item()
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # TensorBoard记录
            if tb and step % 100 == 0:
                tb.add_scalar('train/batch_loss', loss.item(), global_step)
            global_step += 1
        
        # 计算epoch平均loss
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 验证集评估
        val_loss = calc_loss(model, val_loader, loss_function, device, use_seq2seq)
        
        t2 = time.time()
        epoch_time = t2 - t1
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)
        
        # 日志记录
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, time={epoch_time:.2f}s")
        if tb:
            tb.add_scalars('loss', {'train': avg_train_loss, 'val': val_loss}, epoch)
        
        # 终端显示
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")
        
        # 测试评估
        if (epoch + 1) % args.test_epoch == 0:
            bleu, rl = auto_evaluate(model, test_loader, tokenizer, use_seq2seq)
            history['bleu'].append(bleu)
            history['rouge_l'].append(rl)
            
            logger.info(f"Test - BLEU: {bleu:.6f}, Rouge-L: {rl:.6f}")
            if tb:
                tb.add_scalar('test/bleu', bleu, epoch)
                tb.add_scalar('test/rouge_l', rl, epoch)
            
            print(f"  Test BLEU: {bleu:.6f} | Rouge-L: {rl:.6f}")
            
            # 记录预测样例
            demos = predict_demos(model, tokenizer, use_seq2seq, logger)
        
        # 保存最佳模型 + 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            save_model(output_dir / "best_model.bin", model, args, tokenizer)
            logger.info(f"Saved best model with val_loss={val_loss:.6f}")
            print(f"  ✓ Best model saved!")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{args.patience})")
        
        # 早停检查
        if no_improve_count >= args.patience:
            print(f"\n⚠ Early stopping triggered! No improvement for {args.patience} epochs.")
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存训练历史
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # 最终评估
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set:")
    bleu, rl = auto_evaluate(model, test_loader, tokenizer, use_seq2seq)
    print(f"  BLEU: {bleu:.6f} | Rouge-L: {rl:.6f}")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    logger.info(f"Training completed. Best val_loss={best_val_loss:.6f}")
    if tb:
        tb.close()


if __name__ == "__main__":
    run()