# CS224N A2（依存句法分析）你需要做什么 + 推荐顺序（中文）

这份作业的代码骨架已经给好，你的主要工作是把几处 `YOUR CODE HERE / TODO` 补全，让：

- 过渡系统（transition-based parsing）的基本算法能跑通（含 batch 版）
- 前馈神经网络 `ParserModel` 能前向计算
- 训练循环能正确训练并在 dev/test 上评估

> 提示：环境与依赖安装说明仍以 `README.txt` 为准；本文件重点回答“我要改哪里、按什么顺序做、怎么验证”。

---

## 你需要改哪些文件（只改 TODO 区域）

1. `parser_transitions.py`
   - `PartialParse.__init__`：初始化 `stack / buffer / dependencies`
   - `PartialParse.parse_step`：实现三种动作：Shift (`S`)、Left-Arc (`LA`)、Right-Arc (`RA`)
   - `minibatch_parse`：实现 mini-batch 解析算法（按讲义伪代码）

2. `parser_model.py`
   - `__init__`：用 `nn.Parameter` 声明并初始化两层线性变换的参数；创建 dropout
   - `embedding_lookup`：不用 `nn.Embedding`，手写 embedding lookup + reshape
   - `forward`：按讲义完成前向：embedding → hidden（ReLU）→ dropout → logits

3. `run.py`
   - `train`：创建 Adam optimizer 与 CrossEntropyLoss
   - `train_for_epoch`：写训练步：forward → loss → backward → optimizer.step

---

## 推荐完成顺序（照这个做最省时间）

### Step 0：先把环境跑起来

- 用 `README.txt` 的方式创建/激活 conda 环境（`cs224n_a2`）
- 确认能 import torch：
  
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

> Windows：`collect_submission.sh` 里用到 `zip`，可能没有；提交前看本文末尾“打包”。

---

### Step 1：完成 `PartialParse` 的基本动作（最先做，且有单元测试）

修改：`parser_transitions.py`

1) 实现 `PartialParse.__init__`
- `stack`：初始应含 `"ROOT"`
- `buffer`：应是句子词序列的拷贝（不要直接引用/修改原 `sentence`）
- `dependencies`：空列表

2) 实现 `PartialParse.parse_step(transition)`
- `S`：从 buffer 取出第一个词压到 stack
- `LA`：产生 (head, dependent) 并更新 stack（按讲义定义）
- `RA`：产生 (head, dependent) 并更新 stack（按讲义定义）

自测（会跑内置小测试）：

```bash
python parser_transitions.py part_c
```

如果这里不过，先不要往下做。

---

### Step 2：实现 `minibatch_parse`（第二做，同样有测试）

修改：`parser_transitions.py` 中 `minibatch_parse`

目标：给定一堆句子，按 `batch_size` 分组，同时推进多个 `PartialParse`，直到全部完成；并返回与输入句子同顺序的 dependencies 列表。

自测：

```bash
python parser_transitions.py part_d
```

---

### Step 3：实现 `ParserModel`（先跑 sanity check，再训练）

修改：`parser_model.py`

1) `__init__`
- 声明并初始化：
  - `self.embed_to_hidden_weight` / `self.embed_to_hidden_bias`
  - `self.hidden_to_logits_weight` / `self.hidden_to_logits_bias`
- 权重用 `nn.init.xavier_uniform_`，bias 用 `nn.init.uniform_`
- 创建 `self.dropout = nn.Dropout(p=dropout_prob)`

2) `embedding_lookup(w)`
- 输入 `w` 形状：`(batch_size, n_features)`，每个元素是 token id
- 输出 `x` 形状：`(batch_size, n_features * embed_size)`
- 只能用底层张量索引/`index_select`/`gather` 等；不要用 `nn.Embedding`

3) `forward(w)`
- 产出 `logits` 形状：`(batch_size, n_classes)`
- 中间包含 ReLU + dropout

自测（两个 sanity check）：

```bash
python parser_model.py --embedding
python parser_model.py --forward
```

---

### Step 4：补全训练循环并先用 debug 小数据跑通

修改：`run.py`

1) `train(...)`：创建
- `optimizer = optim.Adam(parser.model.parameters(), lr=lr)`
- `loss_func = nn.CrossEntropyLoss()`（默认 reduction 为 mean）

2) `train_for_epoch(...)`：每个 batch
- `logits = parser.model(train_x)`
- `loss = loss_func(logits, train_y)`
- `loss.backward()`
- `optimizer.step()`

先用 debug（小数据集）验证能跑完一个训练流程：

```bash
python run.py -d
```

然后再跑完整数据（会更慢）：

```bash
python run.py
```

输出会在 `results/YYYYMMDD_HHMMSS/` 下保存最优 dev UAS 的模型权重。

---

## 常见注意点（容易踩坑）

- `parser_transitions.PartialParse.__init__`：不要改动传入的 `sentence` 列表本体；需要拷贝。
- `minibatch_parse`：不要对“浅拷贝列表”用 `del` 删除对象（文件里已有解释）。
- `embedding_lookup`：注意 reshape；最终必须是二维 `(batch_size, n_features * embed_size)`。
- `run.py` 的 `-d`：代表 reduced 数据集（更快），用于快速调通流程。

---

## 提交/打包（Windows 友好）

作业脚本 `collect_submission.sh` 在 Windows 可能因为缺少 `zip` 报错。你可以用 PowerShell 手动打包（在 `a2/` 目录下执行）：

```powershell
Compress-Archive -Force -DestinationPath assignment2.zip -Path .\*.py, .\utils
```

生成的 `assignment2.zip` 就是提交包（按课程要求为准）。
