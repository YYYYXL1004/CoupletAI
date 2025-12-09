from typing import List, Tuple
import argparse
from pathlib import Path
import random
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from module import Tokenizer
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CoupletExample(object):
    def __init__(self, seq: List[str], tag: List[str]):
        assert len(seq) == len(tag)
        self.seq = seq
        self.tag = tag


class CoupletFeatures(object):
    def __init__(self, input_ids: List[int], target_ids: List[int]):
        self.input_ids = input_ids
        self.target_ids = target_ids


def read_examples(fdir: Path):
    seqs = []
    tags = []
    with open(fdir / "in.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            seqs.append(line.split())
    with open(fdir / "out.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tags.append(line.split())
    examples = [CoupletExample(seq, tag) for seq, tag in zip(seqs, tags)]
    return examples


def convert_examples_to_features(examples: List[CoupletExample], tokenizer: Tokenizer):
    features = []
    for example in tqdm(examples, desc="creating features"):
        seq_ids = tokenizer.convert_tokens_to_ids(example.seq)
        tag_ids = tokenizer.convert_tokens_to_ids(example.tag)
        features.append(CoupletFeatures(seq_ids, tag_ids))
    return features


def convert_features_to_tensors(features: List[CoupletFeatures], tokenizer: Tokenizer, max_seq_len: int):
    total = len(features)
    input_ids = torch.full((total, max_seq_len),
                           tokenizer.pad_id, dtype=torch.long)
    target_ids = torch.full((total, max_seq_len),
                            tokenizer.pad_id, dtype=torch.long)
    masks = torch.ones(total, max_seq_len, dtype=torch.bool)
    lens = torch.zeros(total, dtype=torch.long)
    for i, f in enumerate(tqdm(features, desc="creating tensors")):
        real_len = min(len(f.input_ids), max_seq_len)
        input_ids[i, :real_len] = torch.tensor(f.input_ids[:real_len])
        target_ids[i, :real_len] = torch.tensor(f.target_ids[:real_len])
        masks[i, :real_len] = 0
        lens[i] = real_len
    return input_ids, masks, lens, target_ids


def create_dataset(fdir: Path, tokenizer: Tokenizer, max_seq_len: int):
    examples = read_examples(fdir)
    features = convert_examples_to_features(examples, tokenizer)
    tensors = convert_features_to_tensors(features, tokenizer, max_seq_len)
    dataset = TensorDataset(*tensors)
    return dataset, examples


def split_train_val(examples: List[CoupletExample], val_ratio: float = 0.1, seed: int = 42):
    """将训练集划分为训练集和验证集"""
    random.seed(seed)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    val_size = int(len(examples) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]
    return train_examples, val_examples


def plot_dataset_split(train_count: int, val_count: int, test_count: int, output_path: Path):
    """绘制数据集划分比例饼图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [train_count, val_count, test_count]
    labels = [f'Train\n{train_count:,}', f'Val\n{val_count:,}', f'Test\n{test_count:,}']
    colors = ['#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.02, 0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    total = sum(sizes)
    ax.text(0, -1.3, f'Total: {total:,} samples', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / 'dataset_split.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"数据集划分图已保存到 {output_path / 'dataset_split.png'}")


def plot_length_distribution(train_examples, val_examples, test_examples, output_path: Path):
    """绘制样本长度分布图"""
    train_lens = [len(ex.seq) for ex in train_examples]
    val_lens = [len(ex.seq) for ex in val_examples]
    test_lens = [len(ex.seq) for ex in test_examples]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    datasets = [('Train', train_lens, '#66b3ff'), 
                ('Val', val_lens, '#99ff99'), 
                ('Test', test_lens, '#ffcc99')]
    
    for ax, (name, lens, color) in zip(axes, datasets):
        ax.hist(lens, bins=30, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(lens), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lens):.1f}')
        ax.axvline(np.median(lens), color='green', linestyle=':', linewidth=2, label=f'Median: {np.median(lens):.1f}')
        ax.set_xlabel('Sequence Length', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name} Set (n={len(lens):,})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        
        # 添加统计信息
        stats_text = f'Min: {min(lens)}\nMax: {max(lens)}\nStd: {np.std(lens):.1f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Sample Length Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'length_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"长度分布图已保存到 {output_path / 'length_distribution.png'}")


def examples_to_dataset(examples: List[CoupletExample], tokenizer: Tokenizer, max_seq_len: int):
    """将examples转换为TensorDataset"""
    features = convert_examples_to_features(examples, tokenizer)
    tensors = convert_features_to_tensors(features, tokenizer, max_seq_len)
    dataset = TensorDataset(*tensors)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='couplet', type=str)
    parser.add_argument("--output", default='dataset', type=str)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float, help="验证集比例")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    vocab_file = input_dir / "vocabs"

    logger.info("creating tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build(vocab_file)

    logger.info("reading examples...")
    # 读取原始训练集和测试集
    all_train_examples = read_examples(input_dir / "train")
    test_examples = read_examples(input_dir / "test")
    
    # 从原始训练集中划分出验证集
    logger.info(f"splitting train/val with ratio {1-args.val_ratio:.1f}/{args.val_ratio:.1f}...")
    train_examples, val_examples = split_train_val(all_train_examples, args.val_ratio, args.seed)
    
    logger.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    # 绘制数据集划分图
    logger.info("plotting dataset statistics...")
    plot_dataset_split(len(train_examples), len(val_examples), len(test_examples), output_dir)
    plot_length_distribution(train_examples, val_examples, test_examples, output_dir)

    # 转换为TensorDataset
    logger.info("creating datasets...")
    train_dataset = examples_to_dataset(train_examples, tokenizer, args.max_seq_len)
    val_dataset = examples_to_dataset(val_examples, tokenizer, args.max_seq_len)
    test_dataset = examples_to_dataset(test_examples, tokenizer, args.max_seq_len)

    logger.info("saving datasets...")
    tokenizer.save_pretrained(output_dir / "vocab.pkl")
    torch.save(train_dataset, output_dir / "train.pkl")
    torch.save(val_dataset, output_dir / "val.pkl")
    torch.save(test_dataset, output_dir / "test.pkl")
    
    logger.info("Done!")
