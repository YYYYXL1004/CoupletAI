# 实验与作业实施计划

## 1. 总体目标
完成从“序列标注”到“Seq2Seq 自回归生成”的架构迁移，并通过多组对比实验和可视化图表，展示模型性能的提升及不同架构的差异，最终形成一份内容详实、数据丰富的实验报告。

---

## 2. 具体实施步骤

### 阶段一：数据准备与分析 (体现基础工作量)
**目标**：不使用现成的 `.pkl`，而是从原始文本重新处理，并进行数据分析。

1.  **自定义数据集划分**：
    *   编写脚本，将原始 `couplet` 数据重新划分为 Train (80%), Validation (10%), Test (10%)。
    *   **图表产出**：绘制数据集的**句子长度分布直方图**，展示上联长度主要集中在什么范围。
2.  **词频统计**：
    *   统计上联和下联的高频字。
    *   **图表产出**：Top-20 高频字词云图或条形图。

### 阶段二：核心模型构建 (Seq2Seq 改造)
**目标**：实现带 Attention 机制的 Encoder-Decoder 架构，这是作业的核心难点。

1.  **Encoder 设计**：
    *   采用 BiLSTM / BiGRU 作为编码器。
    *   输出：Context Vector (隐藏层状态) 和 所有时间步的 Output。
2.  **Decoder 设计 (自回归)**：
    *   输入：前一个时间步的真实字符 (Teacher Forcing 训练时) 或 预测字符 (推理时)。
    *   **Attention 机制**：实现 Bahdanau Attention (Additive) 或 Luong Attention (Dot-product)，让 Decoder 在生成每个字时能“看”到上联的不同部分。
    *   输出：当前步的字符概率分布。
3.  **训练逻辑修改**：
    *   修改 `main.py` 中的 `train` 函数，支持 Seq2Seq 的 Loss 计算（需要 Mask 掉 padding 部分）。

### 阶段三：实验对比 (体现工作量丰富度)
**目标**：通过控制变量法，进行多组实验。

**实验组设计**：
*   **Baseline (基线)**：项目原有的 BiLSTM 序列标注模型。
*   **实验 A (基础 Seq2Seq)**：LSTM Encoder + LSTM Decoder (无 Attention)。
*   **实验 B (架构升级)**：GRU Encoder + GRU Decoder (对比 LSTM 与 GRU 的收敛速度)。
*   **实验 C (最终形态)**：BiLSTM Encoder + LSTM Decoder + **Attention**。

### 阶段四：超参数调优 (体现探究精神)
针对实验 C 的架构，尝试不同的超参数，记录结果：
*   **层数对比**：1层 vs 2层 vs 3层 RNN。
*   **宽度对比**：Hidden Size 128 vs 256 vs 512。

---

## 3. 实验报告图表规划 (为了“好看”)

你的实验报告中需要包含以下关键图表：

1.  **Loss 曲线对比图** (Line Chart)：
    *   横轴 Epoch，纵轴 Loss。
    *   将 Baseline, Seq2Seq(No Attn), Seq2Seq(Attn) 三条线画在同一张图上。
    *   *预期效果*：Seq2Seq(Attn) 的 Loss 下降应该最快且最低。
2.  **评价指标对比图** (Bar Chart)：
    *   对比各模型的 BLEU-2, BLEU-4, Rouge-L 分数。
3.  **生成样例展示表** (Table)：
    *   挑选几个典型的上联，列出不同模型的生成结果，人工点评优劣。
    *   *特意挑选*：挑选长难句，展示 Attention 模型如何解决长距离依赖问题。
4.  **Attention 热力图** (Heatmap) [高分加分项]：
    *   可视化 Attention 权重矩阵。
    *   展示生成下联某个字时，模型主要关注上联的哪个字（例如“天”对“地”，“雨”对“风”）。

---

## 4. 执行 To-Do List

- [ ] **Step 1**: 编写数据划分脚本 `split_data.py`。
- [ ] **Step 2**: 在 `module/` 下新建 `seq2seq.py`，实现 `Encoder` 和 `AttentionDecoder`。
- [ ] **Step 3**: 修改 `main.py`，增加 `--model_type seq2seq` 分支，适配新的输入输出格式。
- [ ] **Step 4**: 跑通第一个 Seq2Seq 训练 demo。
- [ ] **Step 5**: 批量运行实验（改变参数），保存日志。
- [ ] **Step 6**: 编写绘图脚本 `plot_results.py` 生成报告所需图片。
