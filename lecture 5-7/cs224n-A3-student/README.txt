cs224n Assignment 3（学生版）操作说明（README.txt）
================================================

这份文件用来告诉你：在这个仓库里你要实现什么、用什么命令验证、最后怎么打包提交。


0. 你需要完成什么（核心任务）
--------------------------------
你要在 model_solution.py 里把标了 TODO 的模块补全，实现一个“GPT-2 风格、decoder-only”的 Transformer，并让它通过 tests/ 里的快照测试。

主要需要补全的函数（按测试覆盖的顺序）：
1) MLP.forward
2) CausalAttention.forward（带因果 mask 的多头自注意力）
3) DecoderBlock.forward（Pre-LN 残差结构）
4) Transformer.forward（token + positional embedding + 堆叠 decoder blocks + lm head）
5) Transformer.generate（自回归生成若干 token）
6) Transformer.get_loss_on_batch（语言模型 next-token loss）

通过测试后，你的实现应当：
- 形状完全匹配（batch/seq_len/d_model/vocab_size）
- 数值与快照在 atol=1e-5 内一致（除 generate 之外 generate 用的是严格相等）


1. 环境准备（推荐 Conda，Python 3.10）
---------------------------------------
在项目根目录（和 requirements.txt 同级）执行。

1.1 创建并激活环境（Conda）
- 创建：
  conda create -n cs224n-A3 python=3.10
- 激活：
  conda activate cs224n-A3

1.2 安装依赖
- 安装：
  pip install -r requirements.txt

常见问题：
- 首次运行 train.py 会下载 HuggingFace 的数据集/模型/分词器（需要联网）。
- 机器没有 GPU 也能跑测试；训练会更慢。


2. 先跑测试（最重要的验收方式）
---------------------------------
测试使用 snapshots/ 里的输入输出做对齐。

注意：按照仓库自带 README.md 的提示，你需要进入 tests/ 目录再跑 pytest。

2.1 运行全部测试
- 进入 tests 目录：
  cd tests
- 运行：
  pytest

2.2 只跑某个测试（建议你边写边跑）
例如只跑 forward：
  pytest test_student.py::test_forward

测试说明：
- test_mlp / test_attention / test_decoder_block 会分别测试你的 MLP、注意力、decoder block。
- test_forward 测试整个 Transformer.forward。
- test_generate 测试 generate 生成 2 个新 token 的结果是否与快照完全一致。
- test_loss_on_batch 测试 loss 计算是否与快照一致。


3. 代码结构导读（你应该改哪里）
--------------------------------
项目文件概览：
- model_solution.py：你需要补全的主要实现（多处 TODO）。
- utils.py：提供 state_dict_converter，用于把 HuggingFace GPT-2 权重转换到你模型的 key 命名。
- train.py：一个简单训练循环（可用于 sanity check，不是测试必需）。
- tests/：pytest + 快照输入输出。
  - tests/test_student.py：测试入口。
  - tests/snapshots/：测试用的 npy 输入输出，以及 test_model_state_dict.pt。


4. 实现提示（不改测试也能通过的“最低要求”）
---------------------------------------------
下面是实现时最容易踩坑的点（建议你对照写）：

4.1 张量形状约定（非常关键）
- 输入 token ids：x 形状是 [batch, seq_len]
- hidden states：通常是 [batch, seq_len, d_model]
- 注意力分头之后：常见是 [batch, n_heads, seq_len, d_attention]
  其中 d_attention = d_model / n_heads

4.2 CausalAttention.forward（因果 mask）
- 需要实现多头自注意力：Q/K/V 投影 + scaled dot-product attention + 输出投影。
- 因果 mask 的 buffer 已经在 __init__ 里注册了：self.causal_mask
  它的形状是 [1, 1, context_length, context_length]
- 注意：实际 seq_len 可能小于 context_length，记得裁剪 mask。
- mask 的处理通常是：对“未来位置”加上 -inf，然后 softmax。

4.3 DecoderBlock.forward（Pre-LN 残差）
常见结构（GPT-2 风格）：
- x = x + Attention(LN(x))
- x = x + MLP(LN(x))

你这个骨架里有两个 LayerNorm：
- pre_layer_norm
- post_layer_norm

4.4 Transformer.forward
典型步骤：
- token embedding + position embedding
- 依次过 backbone（ModuleList 的多个 DecoderBlock）
- final_layer_norm
- lm_head -> logits: [batch, seq_len, vocab_size]

4.5 generate
- 自回归：每次把当前序列喂进去，取最后一个位置的 logits，选出 next token，再拼回序列。
- 测试使用的是固定快照 + 固定权重，因此你需要与参考实现的“选 token 方式”一致。
  通常是 greedy（argmax），而不是采样。
- 需要注意 context_length：如果序列过长，一般会截断只保留最近的 context_length。

4.6 get_loss_on_batch
- 语言模型 loss：预测下一个 token。
- 常见写法：
  logits = self(input_ids)
  shift_logits = logits[:, :-1, :]
  shift_labels = input_ids[:, 1:]
  用 cross_entropy 计算。


5. 训练（可选：用于跑通端到端、画 loss 曲线）
--------------------------------------------
训练不是通过 tests 的必要条件，但你可以用它 sanity check。

5.1 直接运行
在项目根目录：
  python train.py

train.py 做的事：
- 下载并 tokenization TinyStories 数据集的 1%（首次较慢）
- 把 token 缓存到 ./datasets/...
- 用一个很小的配置（d_model=33 等）跑 100 steps
- 输出 losses_and_grad_norms.png

常见问题：
- 如果你只想先通过单元测试，完全可以不跑训练。


6. 打包提交（submission.zip）
----------------------------
仓库提供了 create_submission.sh，会把以下文件打包进 submission.zip：
- model_solution.py
- utils.py
- train.py

6.1 在 Linux / macOS / WSL / Git Bash
在项目根目录执行：
  bash create_submission.sh

6.2 在纯 Windows PowerShell（没有 bash）
你有三种简单做法：
A) 安装并使用 Git Bash，然后运行上面的 bash 命令。
B) 使用 WSL（Ubuntu 等）进入该目录后运行 bash 命令。
C) 手动打 zip：只把下面 3 个文件压缩为 submission.zip（不要包含 tests/、datasets/ 等）
   - model_solution.py
   - utils.py
   - train.py

验收：
- submission.zip 里只包含上述 3 个文件。


7. 推荐工作流（照着做基本不会走弯路）
------------------------------------
1) 先实现 MLP.forward，跑：pytest test_student.py::test_mlp
2) 再实现 CausalAttention.forward，跑：pytest test_student.py::test_attention
3) 再实现 DecoderBlock.forward，跑：pytest test_student.py::test_decoder_block
4) 实现 Transformer.forward，跑：pytest test_student.py::test_forward
5) 实现 get_loss_on_batch，跑：pytest test_student.py::test_loss_on_batch
6) 实现 generate，跑：pytest test_student.py::test_generate
7) 全部通过后，跑一遍：pytest
8) 打包 submission.zip


8. 需要我接下来帮你做什么？
------------------------
如果你愿意，我可以直接把 model_solution.py 里的 TODO 按照测试快照要求补全，并在本地帮你跑 pytest 验证。
你只需要回复：
- “帮我实现并跑测试”
或者
- “我卡在 XXX（比如 attention mask / generate / loss）”
