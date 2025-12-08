# CoupletAI 项目深度技术解析

## 1. 项目全貌与架构范式
本项目目前采用**序列标注 (Sequence Labeling)** 范式来解决对联生成问题。
核心假设是：**上联与下联字数严格相等，且存在位置对应关系**。
因此，模型并非生成式（Seq2Seq），而是判别式：给定上联序列 $X = (x_1, x_2, ..., x_n)$，预测同长度的下联序列 $Y = (y_1, y_2, ..., y_n)$，其中 $y_i$ 仅依赖于上下文特征 $h_i$。

- **优点**：并行预测，推理速度极快 ($O(1)$ 时间复杂度，非自回归)。
- **缺点**：输出序列内部缺乏依赖性 ($y_i$ 不知道 $y_{i-1}$ 是什么)，容易导致下联内部语义不连贯。

---

## 2. 模块详细分析

### 2.1 数据流与预处理 (`preprocess.py` & `module/tokenizer.py`)

#### Tokenizer 实现
*   **类型**：字符级 Tokenizer (Character-level)。
*   **特殊 Token**：`[PAD]=0`, `[UNK]=1`。没有 `[SOS]` (Start of Sentence) 和 `[EOS]` (End of Sentence)，因为序列标注不需要。
*   **构建方式**：扫描 `vocabs` 文件，按行读取，建立 `token <-> id` 映射。
*   **潜在问题**：直接使用 pickle 保存 Tokenizer 对象实例，如果类定义发生变化，旧的 `.bin` 模型文件将无法加载。

#### 数据集构建 (`CoupletFeatures`)
*   **Input IDs**：上联字符 ID 序列，长度固定为 `max_seq_len` (默认 32)。不足补 `0`。
*   **Target IDs**：下联字符 ID 序列，同样补 `0`。
*   **Masks**：Bool Tensor，有效字符为 `1`，Padding 为 `0`。
*   **Lens**：记录实际有效长度。
*   **存储格式**：`TensorDataset` 被直接 `torch.save` 为 `.pkl` 文件。

### 2.2 模型层级细节 (`module/model.py`)

所有模型均继承自 `nn.Module`，且遵循以下 Input/Output 协议：
*   **Input**: `(batch_size, seq_len)`
*   **Output**: `(batch_size, seq_len, vocab_size)` -> **Logits** (未经过 Softmax)

#### 核心模型：`BiLSTM`
1.  **Embedding**: `(batch, seq) -> (batch, seq, embed_dim)`
2.  **Dropout**: `embed_dropout`
3.  **Encoder**: `nn.LSTM(..., bidirectional=True)`
    *   输出维度: `(batch, seq, hidden_dim * 2)` (双向拼接)
    *   **注意**：这里直接使用了 LSTM 的输出序列，没有取最后一帧的 hidden state。
4.  **Projection**: `nn.Linear(hidden_dim, embed_dim)` -> *这里有个维度压缩，通常是直接映射到 vocab_size，但它多加了一层*。
    *   代码逻辑：`x = self.linear(x)` -> `(batch, seq, embed_dim)`
    *   最后输出：`torch.matmul(x, self.embedding.weight.t())` -> **Weight Tying (权重共享)**。
    *   **关键点**：输出层的 Linear 权重与 Input Embedding 权重共享。这是一种常见的正则化手段，能减少参数量。

#### 其他模型变体
*   `Transformer`: 仅使用了 `nn.TransformerEncoder`，没有 Decoder。位置编码使用的是标准的 `nn.Embedding` 学习出的 Positional Embedding，而不是正弦波编码。
*   `BiLSTMAttn`: 在 BiLSTM 输出后加了 Self-Attention (`nn.MultiheadAttention`)。
*   `BiLSTMCNN`: 在 BiLSTM 后加了 1D 卷积，意图捕捉局部特征。

### 2.3 训练与评估 (`main.py` & `module/metric.py`)

#### 训练循环
*   **Loss Function**: `nn.CrossEntropyLoss(ignore_index=0)`。自动忽略 Pad 部分的 Loss，无需手动 Mask。
*   **Optimizer**: `Adam`。
*   **Scheduler**: `ReduceLROnPlateau` (当 Loss 不下降时减小 LR)。
*   **Mixed Precision**: 支持 NVIDIA Apex (`fp16`)，这是一个较老的混合精度库，现在 PyTorch 推荐用 `torch.cuda.amp`。

#### 评估指标
*   **BLEU**: 使用 `nltk.translate.bleu_score`，计算生成序列与参考序列的 N-gram 重合度。
*   **Rouge-L**: 基于最长公共子序列 (LCS) 的召回率指标。
*   **Evaluation 逻辑**: `auto_evaluate` 函数中，对 Test Set 进行推理，取 `argmax` 得到预测序列，然后计算 BLEU/Rouge。

### 2.4 推理逻辑 (`clidemo.py`)
*   **加载方式**：`torch.load` 读取整个字典，通过 `init_model_by_key` 反射构建模型。
*   **解码策略**：`Greedy Search` (贪婪搜索)。
    *   `logits.argmax(dim=-1)`。
    *   没有 Beam Search，也没有采样 (Sampling)。

---

## 3. 面向作业的改造痛点分析

要将此项目改为 Seq2Seq (Encoder-Decoder)，必须进行以下“大手术”：

1.  **数据预处理 (Preprocess)**：
    *   **必须添加 Special Tokens**：Seq2Seq 的 Decoder 需要 `[SOS]` (Start) 作为起始输入，需要 `[EOS]` (End) 作为生成结束标志。
    *   目前的 Tokenizer 没有这两个 token，需要修改 `vocabs` 文件或在代码中动态添加。

2.  **模型架构 (Model)**：
    *   目前的 `BiLSTM` 只有 Encoder。
    *   **新增 Decoder**：需要实现一个单向 LSTM/GRU，初始状态继承自 Encoder 的最终状态。
    *   **Attention 接口**：Decoder 的每一步都需要查询 Encoder 的 Output。

3.  **训练逻辑 (Train)**：
    *   **Teacher Forcing**：训练时，Decoder 的输入应该是 Ground Truth (右移一位)。
    *   **Loss 计算**：目前的 CrossEntropy 依然可用，但 Input 和 Target 的对齐方式变了。

4.  **推理逻辑 (Inference)**：
    *   原来的 `argmax` 并行预测失效。
    *   需要写一个 `loop`，不断将上一时刻的输出作为当前时刻的输入，直到生成 `[EOS]`。
