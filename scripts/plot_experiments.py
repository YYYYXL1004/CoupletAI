"""
实验结果可视化脚本
生成：Loss曲线、BLEU/Rouge-L柱状图、消融实验图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验配置
EXPERIMENTS = {
    'Exp1_BiLSTM_L1_H256': {'name': 'BiLSTM', 'color': '#1f77b4'},
    'Exp2_Seq2Seq_L1_H256': {'name': 'Seq2Seq L1', 'color': '#ff7f0e'},
    'Exp3_Seq2Seq_L2_H256': {'name': 'Seq2Seq L2', 'color': '#2ca02c'},
    'Exp4_Seq2Seq_L2_H512': {'name': 'Seq2Seq L2 H512', 'color': '#d62728'},
    'Exp5_Seq2Seq_Attn_L2_H256': {'name': 'Seq2Seq+Attn', 'color': '#9467bd'},
    'Exp6_Transformer_L1_H256': {'name': 'Transformer', 'color': '#8c564b'},
    'Exp7_BiLSTMAttn_L1_H256': {'name': 'BiLSTMAttn', 'color': '#e377c2'},
}

OUTPUT_DIR = 'output'
FIGURE_DIR = 'figures'


def load_all_histories():
    """加载所有实验的history.json"""
    histories = {}
    for exp_name in EXPERIMENTS:
        path = os.path.join(OUTPUT_DIR, exp_name, 'history.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                histories[exp_name] = json.load(f)
    return histories


def plot_loss_curves(histories):
    """绘制训练/验证Loss曲线对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练Loss
    ax1 = axes[0]
    for exp_name, hist in histories.items():
        cfg = EXPERIMENTS[exp_name]
        epochs = range(1, len(hist['train_loss']) + 1)
        ax1.plot(epochs, hist['train_loss'], label=cfg['name'], color=cfg['color'], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 验证Loss
    ax2 = axes[1]
    for exp_name, hist in histories.items():
        cfg = EXPERIMENTS[exp_name]
        epochs = range(1, len(hist['val_loss']) + 1)
        ax2.plot(epochs, hist['val_loss'], label=cfg['name'], color=cfg['color'], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {FIGURE_DIR}/loss_curves.png")


def plot_metrics_bar(histories):
    """绘制BLEU/Rouge-L柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [EXPERIMENTS[k]['name'] for k in histories.keys()]
    colors = [EXPERIMENTS[k]['color'] for k in histories.keys()]
    
    # 取最后一个epoch的指标
    bleu_scores = [hist['bleu'][-1] * 100 for hist in histories.values()]
    rouge_scores = [hist['rouge_l'][-1] * 100 for hist in histories.values()]
    
    # BLEU柱状图
    ax1 = axes[0]
    bars1 = ax1.bar(names, bleu_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('BLEU Score (%)', fontsize=12)
    ax1.set_title('BLEU Score Comparison', fontsize=14)
    ax1.set_ylim(0, max(bleu_scores) * 1.2)
    for bar, score in zip(bars1, bleu_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{score:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.tick_params(axis='x', rotation=15)
    
    # Rouge-L柱状图
    ax2 = axes[1]
    bars2 = ax2.bar(names, rouge_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Rouge-L Score (%)', fontsize=12)
    ax2.set_title('Rouge-L Score Comparison', fontsize=14)
    ax2.set_ylim(0, max(rouge_scores) * 1.2)
    for bar, score in zip(bars2, rouge_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{score:.2f}%', ha='center', va='bottom', fontsize=10)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {FIGURE_DIR}/metrics_comparison.png")


def plot_ablation_study(histories):
    """绘制消融实验图（层数和隐藏维度的影响）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 层数影响 (Exp2 vs Exp3: 1层 vs 2层)
    ax1 = axes[0]
    if 'Exp2_Seq2Seq_L1_H256' in histories and 'Exp3_Seq2Seq_L2_H256' in histories:
        layers = [1, 2]
        bleu_by_layer = [
            histories['Exp2_Seq2Seq_L1_H256']['bleu'][-1] * 100,
            histories['Exp3_Seq2Seq_L2_H256']['bleu'][-1] * 100,
        ]
        rouge_by_layer = [
            histories['Exp2_Seq2Seq_L1_H256']['rouge_l'][-1] * 100,
            histories['Exp3_Seq2Seq_L2_H256']['rouge_l'][-1] * 100,
        ]
        ax1.plot(layers, bleu_by_layer, 'o-', label='BLEU', color='#1f77b4', linewidth=2, markersize=10)
        ax1.plot(layers, rouge_by_layer, 's-', label='Rouge-L', color='#ff7f0e', linewidth=2, markersize=10)
        ax1.set_xlabel('Number of Layers', fontsize=12)
        ax1.set_ylabel('Score (%)', fontsize=12)
        ax1.set_title('Effect of Layer Depth (H=256)', fontsize=14)
        ax1.set_xticks(layers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Effect of Layer Depth (H=256)', fontsize=14)
    
    # 隐藏维度影响 (Exp3 vs Exp4: 256 vs 512)
    ax2 = axes[1]
    if 'Exp3_Seq2Seq_L2_H256' in histories and 'Exp4_Seq2Seq_L2_H512' in histories:
        hidden_dims = [256, 512]
        bleu_by_hidden = [
            histories['Exp3_Seq2Seq_L2_H256']['bleu'][-1] * 100,
            histories['Exp4_Seq2Seq_L2_H512']['bleu'][-1] * 100,
        ]
        rouge_by_hidden = [
            histories['Exp3_Seq2Seq_L2_H256']['rouge_l'][-1] * 100,
            histories['Exp4_Seq2Seq_L2_H512']['rouge_l'][-1] * 100,
        ]
        ax2.plot(hidden_dims, bleu_by_hidden, 'o-', label='BLEU', color='#1f77b4', linewidth=2, markersize=10)
        ax2.plot(hidden_dims, rouge_by_hidden, 's-', label='Rouge-L', color='#ff7f0e', linewidth=2, markersize=10)
        ax2.set_xlabel('Hidden Dimension', fontsize=12)
        ax2.set_ylabel('Score (%)', fontsize=12)
        ax2.set_title('Effect of Hidden Size (L=2)', fontsize=14)
        ax2.set_xticks(hidden_dims)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Effect of Hidden Size (L=2)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'ablation_study.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {FIGURE_DIR}/ablation_study.png")


def plot_seq_labeling_comparison(histories):
    """绘制序列标注模型对比图（BiLSTM vs Transformer vs BiLSTMAttn）"""
    seq_labeling_exps = ['Exp1_BiLSTM_L1_H256', 'Exp6_Transformer_L1_H256', 'Exp7_BiLSTMAttn_L1_H256']
    available = {k: v for k, v in histories.items() if k in seq_labeling_exps}
    
    if len(available) < 2:
        print("⚠ 序列标注对比图需要至少2个实验数据，跳过")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [EXPERIMENTS[k]['name'] for k in available.keys()]
    colors = [EXPERIMENTS[k]['color'] for k in available.keys()]
    bleu_scores = [hist['bleu'][-1] * 100 for hist in available.values()]
    rouge_scores = [hist['rouge_l'][-1] * 100 for hist in available.values()]
    
    # BLEU
    ax1 = axes[0]
    bars1 = ax1.bar(names, bleu_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('BLEU Score (%)', fontsize=12)
    ax1.set_title('Sequence Labeling Models - BLEU', fontsize=14)
    ax1.set_ylim(0, max(bleu_scores) * 1.3 if bleu_scores else 1)
    for bar, score in zip(bars1, bleu_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{score:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Rouge-L
    ax2 = axes[1]
    bars2 = ax2.bar(names, rouge_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Rouge-L Score (%)', fontsize=12)
    ax2.set_title('Sequence Labeling Models - Rouge-L', fontsize=14)
    ax2.set_ylim(0, max(rouge_scores) * 1.2 if rouge_scores else 1)
    for bar, score in zip(bars2, rouge_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{score:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'seq_labeling_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {FIGURE_DIR}/seq_labeling_comparison.png")


def main():
    # 创建输出目录
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # 加载数据
    print("加载实验数据...")
    histories = load_all_histories()
    print(f"已加载 {len(histories)} 个实验结果\n")
    
    # 绘制图表
    print("生成图表...")
    plot_loss_curves(histories)
    plot_metrics_bar(histories)
    plot_ablation_study(histories)
    plot_seq_labeling_comparison(histories)
    
    print(f"\n✅ 所有图表已保存到 {FIGURE_DIR}/ 目录")


if __name__ == '__main__':
    main()
