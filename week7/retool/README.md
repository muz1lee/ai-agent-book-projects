# ReTool：使用多轮对话和代码沙箱提升大语言模型数学推理能力

## 概述

ReTool 是一种创新的强化学习方法，旨在通过多轮对话和代码沙箱执行来显著提升大语言模型在数学推理任务上的表现。该方法的核心思想是让模型学会使用工具（特别是代码执行环境）来辅助数学问题的求解，而不是仅依赖于语言模型自身的推理能力。通过有监督微调（SFT）和强化学习（RL）两个阶段的训练，ReTool 能够教会模型何时以及如何调用代码沙箱来验证计算结果、测试假设或探索问题空间。

本文档详细记录了 ReTool 方法的完整复现步骤，包括环境配置、模型训练以及评估过程。该复现基于 verl 框架，使用 Qwen2.5-32B-Instruct 作为基础模型，并在 AIME 2024 数学竞赛数据集上进行训练。整个训练过程分为两个主要阶段：首先通过有监督微调使模型学会基本的工具使用模式，然后通过强化学习进一步优化模型在实际问题求解中的表现。

## 硬件与软件要求

### 硬件配置

复现 ReTool 需要强大的 GPU 计算资源。推荐的硬件配置包括以下两种方案：

第一种方案是使用一台配备 8 卡 H200 GPU 的服务器。H200 GPU 具有较大的显存容量和强大的计算能力，能够高效地进行大规模模型的训练。单台服务器的配置简化了分布式训练的复杂性，特别适合初次尝试复现的研究者。

第二种方案是使用 2 台 8 卡 A100 或 H100 GPU 的服务器。这种配置提供了更大的总计算能力，但需要配置多机分布式训练。如果选择这种方案，建议使用原始的 verl 框架而非修改版本。

### 软件环境

推荐使用以下软件环境组合以获得最佳的兼容性和性能：

- **CUDA 版本**：12.6.2
- **操作系统**：Ubuntu 24.04 LTS
- **Python 版本**：3.13

这些版本的选择基于对最新深度学习框架的支持以及稳定性考虑。CUDA 12.6.2 提供了对最新 GPU 架构的优化支持，Ubuntu 24.04 是长期支持版本，确保了系统的稳定性，而 Python 3.13 则提供了最新的语言特性和性能改进。

## 环境搭建

### 下载 verl 框架

verl 是一个高效的强化学习框架，专门为大语言模型的 RLHF（Reinforcement Learning from Human Feedback）训练而设计。对于单台 8 卡 H200 服务器的配置，建议使用经过修改的版本：

```bash
git clone https://github.com/bojieli/verl
```

该修改版本针对单台 8 卡 H200 的配置进行了优化。如果使用多机配置，则应该使用原始的 verl 框架：

```bash
git clone https://github.com/volcengine/verl/
```

### 安装 Miniconda

Miniconda 是一个轻量级的 Python 环境管理工具，能够方便地创建和管理独立的 Python 环境，避免依赖冲突。安装步骤如下：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中按照提示完成配置。安装完成后，建议重新启动终端以使环境变量生效。

### 创建 Conda 环境

为了避免与系统中其他项目的依赖冲突，创建一个专门用于 ReTool 训练的 Conda 环境：

```bash
conda create -n verl python==3.13
conda activate verl
```

激活环境后，所有后续的包安装和命令执行都将在这个隔离的环境中进行。

### 安装依赖包

进入 verl 目录并安装所需的依赖包。这个过程包括安装基础依赖、CUDA 相关依赖以及 verl 本身：

```bash
cd verl
pip install -r requirements.txt
pip install -r requirements-cuda.txt
pip install -e .
```

其中 `requirements.txt` 包含了基础的 Python 包依赖，`requirements-cuda.txt` 包含了 CUDA 相关的深度学习框架和库，而 `pip install -e .` 则以可编辑模式安装 verl 框架本身，方便后续的开发和调试。

### 下载基础模型

ReTool 使用 Qwen2.5-32B-Instruct 作为基础模型。这是一个经过指令微调的大语言模型，具有较强的指令遵循能力和推理能力。首先创建模型存储目录，然后使用 Hugging Face CLI 下载模型：

```bash
mkdir -p /root/verl/recipe/retool/model/
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
    --local-dir /root/verl/recipe/retool/model/Qwen2.5-32B-Instruct \
    --local-dir-use-symlinks False
```

参数 `--local-dir-use-symlinks False` 确保文件被完整复制而非使用符号链接，这在某些训练场景下更加稳定可靠。

### 下载训练数据

ReTool 的有监督微调阶段使用 AIME 2024 数据集。首先需要运行预处理脚本，然后下载数据集：

```bash
python3 recipe/retool/retool_sft_preprocess.py
huggingface-cli download --repo-type dataset --resume-download \
    BytedTsinghua-SIA/AIME-2024 \
    --local-dir /dataset/BytedTsinghua-SIA/AIME_2024
```

预处理脚本会准备数据集的格式，使其符合训练框架的输入要求。参数 `--resume-download` 允许在网络中断时继续下载，提高了下载的健壮性。

## 有监督微调（SFT）

### 启动训练

有监督微调是 ReTool 训练的第一个阶段，目的是让模型学会基本的工具使用模式和多轮对话格式。进入配方目录并执行训练脚本：

```bash
cd recipe/retool
bash run_qwen2-32b_sft.sh
```

### 配置 Wandb

训练开始时，系统会提示配置 Weights & Biases（wandb），这是一个流行的机器学习实验跟踪工具。首先需要在 wandb.ai 注册账号，然后按照提示操作：

1. 选择"Use an existing W&B account"（选项 2）
2. 访问 https://wandb.ai/authorize 获取 API key
3. 将 API key 粘贴到终端提示符中

配置完成后，训练过程的所有指标都会自动上传到 wandb 平台，可以通过网页界面实时监控训练进度和模型性能。

### 训练过程解析

训练开始后，系统会显示详细的配置信息和训练参数。根据数据集大小和批次设置，训练过程包括 6 个 epoch，每个 epoch 62 个 steps，总共 372 个训练步骤。

训练配置的关键参数包括：

- **批次大小**：训练批次大小为 16，每个 GPU 的微批次大小为 4
- **序列长度**：最大序列长度为 16384，支持长文本的多轮对话
- **优化器配置**：学习率为 1e-5，使用 Adam 优化器，权重衰减为 0.01
- **并行策略**：使用 Ulysses 序列并行，并行大小为 4，有效处理长序列
- **模型策略**：使用 FSDP（Fully Sharded Data Parallel）进行分布式训练，并启用梯度检查点以节省显存

训练过程中会输出每个步骤的损失值、学习率和执行时间。典型的训练日志如下：

```
step:1 - train/loss:0.8078852891921997 - train/lr(1e-3):0.0002702702702702703 - train/time(s):14.796027898788452
step:2 - train/loss:0.7787683010101318 - train/lr(1e-3):0.0005405405405405405 - train/time(s):7.293778896331787
step:3 - train/loss:0.7899439334869385 - train/lr(1e-3):0.0008108108108108109 - train/time(s):6.083798885345459
```

可以观察到，初始步骤由于各种初始化操作耗时较长（约 15 秒），但后续步骤稳定在每步 6-7 秒左右。随着训练的进行，损失值逐渐下降，表明模型正在学习任务。

到第 3 个 epoch 时，损失已经显著降低：

```
step:127 - train/loss:0.1943996697664261 - train/lr(1e-3):0.00832235736719411 - train/time(s):6.062393665313721
step:128 - train/loss:0.1821298599243164 - train/lr(1e-3):0.008287170670328432 - train/time(s):6.20814323425293
```

学习率采用 cosine 调度策略，在训练过程中逐渐衰减，这有助于模型在训练后期更加稳定地收敛。


### 训练时长估算

根据实际测试，在 8 卡 H200 GPU 配置下，平均每个 step 耗时约 7 秒。完成全部 372 个 steps 大约需要 45 分钟。实际时间可能因 GPU 配置有所不同。

训练完成后，终端会输出总结信息，可以点击 wandb 链接来查看训练过程的详细信息。

```
Total time for train steps: 2627.97s
Final validation metrics: {'val/loss': 0.019425522536039352}
Epoch 6/6:  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎  | 61/62 [09:53<00:09,  9.73s/it]
wandb:
wandb: Run history:
wandb:     train/loss █▇▇▇▅▅▅▅▅▄▄▄▄▄▂▂▂▂▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: train/lr(1e-3) ▄▅▆▆██████▇▇▇▇▆▆▆▆▆▆▄▄▄▄▃▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁
wandb:  train/time(s) ▅▃▆▆▃▃▂█▁▃▆▆▁▆▄▅▂▃▄▂▅▁▄▅▄▇▄▁▂▂▂▂▄▂▃▄▄█▁▅
wandb:       val/loss ▁
wandb:
wandb: Run summary:
wandb:     train/loss 0.01913
wandb: train/lr(1e-3) 0
wandb:  train/time(s) 6.46921
wandb:       val/loss 0.01943
wandb:
wandb: 🚀 View run multiturn-sft-qwen-2.5-32b-instruct at: https://wandb.ai/bojieli-pine-ai/boj-multiturn-sft/runs/7zndjepf
wandb: ⭐️ View project at: https://wandb.ai/bojieli-pine-ai/boj-multiturn-sft
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
```

训练完成后，模型检查点会保存在指定的目录中（默认为 `/root/verl/recipe/retool/checkpoint/multiturn-sft-qwen-2.5-32b-instruct`），用于后续的强化学习阶段。

### SFT 训练配方参数详解

理解训练脚本的参数配置对于调整训练过程和优化性能至关重要。SFT 训练脚本通过 torchrun 启动分布式训练，其中 nnodes 和 nproc_per_node 参数分别指定了节点数量和每个节点的进程数。对于单台 8 卡 H200 服务器，这两个参数分别设置为 1 和 8，意味着在单个节点上使用全部 8 张 GPU 进行训练。standalone 选项表示不需要配置主节点地址，简化了单机训练的启动流程。

数据相关的参数决定了模型的输入处理方式。train_files 和 val_files 参数指定训练和验证数据的路径，这里使用的是 ReTool-SFT 数据集的 parquet 格式文件。max_length 参数设置为 16384，这是一个相当长的序列长度，允许模型处理包含多轮对话和代码执行结果的复杂交互序列。train_batch_size 设置为 32，表示每个训练步骤处理 32 个样本，而 micro_batch_size_per_gpu 设置为 4，意味着每张 GPU 实际一次处理 4 个样本，通过梯度累积实现更大的有效批次大小。这种设计在有限的 GPU 显存下平衡了训练稳定性和效率。

多轮对话的配置是 ReTool 训练的关键特性。data.multiturn.enable 参数启用多轮对话模式，messages_key 和 tools_key 参数分别指定数据集中对话消息和工具定义的字段名称。这使得模型能够学习在对话过程中适时调用工具的能力。模型通过观察示例数据中的消息序列和工具调用模式，逐渐理解何时需要使用工具、如何构造工具调用请求以及如何解释工具返回的结果。

模型和训练策略参数控制了模型的加载和优化方式。partial_pretrain 参数指向 Qwen2.5-32B-Instruct 基础模型的路径，模型将从这个预训练检查点开始微调。strategy 参数设置为 fsdp（Fully Sharded Data Parallel），这是一种先进的分布式训练策略，将模型参数、梯度和优化器状态分片到多个 GPU 上，大幅降低单卡显存需求。

序列并行和优化参数进一步提升了训练效率。ulysses_sequence_parallel_size 设置为 4，表示使用 Ulysses 序列并行算法将长序列切分到 4 张 GPU 上并行处理。这对于处理 16384 长度的序列至关重要，避免了单卡显存的瓶颈。use_remove_padding 参数启用动态填充移除优化，只对实际的有效 token 进行计算，跳过填充 token，显著提升了训练效率，特别是在处理变长序列时效果明显。

实验管理参数帮助组织和追踪训练过程。project_name 和 experiment_name 用于在 wandb 平台上标识训练任务，便于后续查看和对比不同实验的结果。default_local_dir 指定检查点的本地保存路径，训练过程中会定期保存模型状态。logger 参数设置为同时使用控制台和 wandb 两种日志方式，确保既能实时观察训练进度，又能通过 wandb 网页界面进行深入分析。total_epochs 设置为 6，表示完整遍历训练数据集 6 次。

## 强化学习环境：SandboxFusion

### 环境简介

SandboxFusion 是 ReTool 强化学习阶段使用的代码执行沙箱环境。它提供了一个安全、隔离的 Python 代码执行环境，允许模型在训练过程中实际执行代码并获得执行结果的反馈。这种交互式的学习方式是 ReTool 能够有效提升模型工具使用能力的关键。

### 安装步骤

下载 SandboxFusion 的修改版本，该版本支持在本地运行 128 个并行 worker，满足大规模训练的需求：

```bash
git clone https://github.com/bojieli/SandboxFusion
cd SandboxFusion/
```

为 SandboxFusion 创建独立的 Conda 环境。注意这里使用 Python 3.12 而非 3.13，这是为了与 SandboxFusion 的依赖保持最佳兼容性：

```bash
conda create -n sandbox python==3.12
conda activate sandbox
```

使用 Poetry 安装依赖。Poetry 是一个现代化的 Python 依赖管理工具，能够更好地处理复杂的依赖关系：

```bash
pip install poetry
poetry install
```

新建一个用于代码执行沙盒的 conda 环境，并安装常见的科学计算、数学相关库。

```bash
conda create -n sandbox-runtime -y python=3.11
conda activate sandbox-runtime
pip install -r ./requirements.txt --ignore-requires-python

# for NaturalCodeBench python problem 29
python -c "import nltk; nltk.download('punkt')"

# for CIBench nltk problems
python -c "import nltk; nltk.download('stopwords')"
```

启动 SandboxFusion 服务（默认并行 128 个进程，如果机器配置较低，可修改 Makefile）。该服务会监听来自训练进程的代码执行请求：

```bash
make run-online
```

为了确保服务能够正常执行科学计算代码，建议使用如下命令测试：

```bash
curl 'http://localhost:8080/run_code' \
  -H 'Content-Type: application/json' \
  --data-raw '{"code": "import sympy\n\n# Define symbolic variables\nx, y = sympy.symbols(\"x y\")\n\n# Basic algebra\nexpr = x**2 + 2*x + 1\nfactored = sympy.factor(expr)\nprint(f\"Expression: {expr}\")\nprint(f\"Factored: {factored}\")\n\n# Solve equation\nsolution = sympy.solve(x**2 - 4, x)\nprint(f\"Solution to x^2 - 4 = 0: {solution}\")\n\n# Calculus\nderivative = sympy.diff(x**3 + 2*x**2 + x, x)\nprint(f\"Derivative of x^3 + 2x^2 + x: {derivative}\")\n\n# Integration\nintegral = sympy.integrate(x**2, x)\nprint(f\"Integral of x^2: {integral}\")\n\nprint(\"\\nSympy test completed successfully!\")", "language": "python"}'
```

如果服务运行正常，将输出下述结果。

```
{"status":"Success","message":"","compile_result":null,"run_result":{"status":"Finished","execution_time":0.2835578918457031,"return_code":0,"stdout":"Expression: x**2 + 2*x + 1\nFactored: (x + 1)**2\nSolution to x^2 - 4 = 0: [-2, 2]\nDerivative of x^3 + 2x^2 + x: 3*x**2 + 4*x + 1\nIntegral of x^2: x**3/3\n\nSympy test completed successfully!\n","stderr":""},"executor_pod_name":null,"files":{}}
```

服务正常启动后，强化学习训练进程就可以通过 API 调用 SandboxFusion 来执行模型生成的代码，并将执行结果作为奖励信号反馈给训练过程。

## 强化学习训练（RL）

### 准备 SFT 模型检查点

完成有监督微调后，需要将 FSDP 格式的检查点转换为 Hugging Face 标准格式，以便在强化学习阶段使用。verl 框架提供了模型合并工具来完成这一转换过程。SFT 训练共 372 个步骤，使用最后一个检查点进行转换：

```bash
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /root/verl/recipe/retool/checkpoint/multiturn-sft-qwen-2.5-32b-instruct/global_step_372 \
    --target_dir /root/verl/recipe/retool/checkpoint/multiturn-sft-qwen-2.5-32b-instruct/global_step_372/huggingface
```

这个转换过程会将分布式训练中分片保存的模型权重合并为单一的模型文件，并转换为 Hugging Face Transformers 库的标准格式。转换后的模型将保存在指定的 `huggingface` 子目录中，可以直接用于后续的强化学习训练或推理评估。需要注意的是，这个过程需要足够的磁盘空间来存储转换后的模型文件。

### ReTool 训练原理

在深入 RL 训练步骤之前，有必要理解 ReTool 方法的核心原理。根据 [ReTool 论文](https://arxiv.org/pdf/2504.11536)，ReTool 的创新之处在于将工具使用能力整合到大语言模型的推理过程中，使模型能够像人类数学家一样利用计算工具来辅助问题求解。传统的推理模型（如 DeepSeek R1 和 OpenAI o1）虽然在纯文本推理任务上表现出色，但在需要精确数值计算或符号操作的场景中仍然存在明显局限。文本推理过程容易产生累积误差和计算错误，而代码解释器通过提供形式化和可执行的接口，能够实现精确的数值验证，显著减少了这类问题。

ReTool 的训练流程分为两个关键阶段。在有监督微调（SFT）阶段，模型通过学习高质量的代码增强推理轨迹来掌握基本的工具调用模式。数据构建流程首先从开源数据集（如 OpenThoughts）收集数学推理数据，经过人类专家和 DeepSeek-R1 的双重验证筛选后，使用结构化提示模板将文本推理过程转换为代码集成的推理数据。这个转换过程会将原始思考过程中可以通过代码执行获益的手工计算步骤替换为相应的代码片段及其执行结果。转换后的数据经过格式验证和答案验证两个阶段，确保语法一致性和结果正确性。SFT 阶段使用 swordfaith/ReTool-SFT-multi-turn 数据集，该数据集包含 2000 个数学问题及其详细的多轮对话解答过程，每个样本都标注了 tool_call 属性。通过这种监督学习，模型建立了工具使用的基础认知框架。

强化学习阶段则采用 DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）算法，这是专门为大规模长思维链 RL 场景设计的优化方法。根据 [DAPO 论文](https://arxiv.org/pdf/2503.14476)，该算法在 Qwen2.5-32B 模型上达到了 AIME 2024 的 50 分准确率，超越了 DeepSeek-R1-Zero-Qwen-32B 的 47 分，且仅使用了后者 50% 的训练步数。DAPO 在 GRPO（Group Relative Policy Optimization）的基础上引入了四项关键技术创新，专门解决长思维链 RL 训练中的熵崩溃、奖励噪声和训练不稳定等问题。

第一项技术是 Clip-Higher 策略，旨在促进系统多样性并避免熵崩溃。传统 PPO 使用对称的裁剪范围（如 1-ε 到 1+ε），但在长思维链场景中，这种对称裁剪会导致模型过早收敛到确定性策略，丧失探索能力。Clip-Higher 通过使用不对称的裁剪区间（clip_ratio_low=0.2, clip_ratio_high=0.28），为向上的策略更新提供更大的空间，鼓励模型探索高奖励的响应模式，同时严格限制向下的更新以防止性能退化。这种设计在保持训练稳定性的同时，有效维持了策略的探索性，避免了熵过早崩溃到接近零的状态。

第二项技术是动态采样（Dynamic Sampling），用于提高训练效率和稳定性。在标准 GRPO 中，每个问题固定生成 G 个响应用于相对优势估计。动态采样根据训练进程和奖励分布动态调整每个问题的采样数量，在训练初期使用更多采样以获得稳定的优势估计，在训练后期当策略趋于稳定时减少采样以提高效率。第三项技术是 Token-Level Policy Gradient Loss，这在长思维链场景中至关重要。传统方法对整个序列使用单一的奖励信号，而 DAPO 将损失函数在 token 级别进行归一化，确保长短序列获得公平的梯度权重，避免超长序列主导训练过程。第四项技术是 Overlong Reward Shaping，通过对超长响应施加轻微的长度惩罚来减少奖励噪声。这个机制不是硬性限制长度，而是在奖励函数中引入温和的负反馈，引导模型在保证推理质量的前提下生成更简洁高效的响应。

ReTool 的核心创新在于支持交织的实时代码执行的 rollout 机制。在生成过程中，策略模型与代码沙箱协作，动态产生混合内容——包括文本推理、代码片段和实时的解释器反馈。具体而言，模型使用特定的标签（`<code></code>`）标记生成的代码边界。当检测到代码终止触发器（`</code>`）时，生成暂停，解析后的代码被发送到沙箱执行。沙箱的输出（成功结果或错误消息）被填充在 `<interpreter></interpreter>` 标签中并反馈给模型，模型继续生成推理轨迹，最终产生形如 `[文本₁ ⊕ 代码₁ ⊕ 反馈₁ ⊕ ... ⊕ 答案]` 的混合推理序列。

奖励设计采用了极简主义的准则驱动方法。ReTool 使用基于规则的准确性奖励，当预测答案与标准答案等价时获得 +1 奖励，否则获得 -1 奖励。这种简化的奖励设计旨在缓解奖励欺骗问题，并在仅基于结果反馈的情况下促进更多样化的问题解决行为。论文中明确指出不考虑代码可执行性奖励，而是让模型通过结果反馈自主学习何时以及如何调用工具。这种设计理念的关键在于相信模型能够通过足够的探索，自主发现最优的工具调用模式，而不需要人为设定工具使用的先验规则。

实验结果充分验证了 ReTool 方法的有效性。在 AIME 2024 数据集上，基于 Qwen2.5-32B-Instruct 的 ReTool 模型仅用 400 个训练步骤就达到了 67.0% 的准确率，而文本 RL 基线即使经过 1080 个训练步骤也只达到 40.0% 的准确率。这一显著差距表明，显式地将工具使用建模为决策过程的一部分，不仅推动了模型推理能力的边界，还大幅提升了训练效率。在扩展设置下，ReTool-32B 达到了 72.5% 的准确率，超越 OpenAI 的 o1-preview 模型 27.9 个百分点。

ReTool 方法的另一个重要特点是其多轮交互机制和涌现行为。与传统的单次生成不同，ReTool 允许模型进行多轮思考和工具调用。模型可以先提出假设，然后通过代码验证假设的正确性，根据验证结果调整推理方向，再进行下一轮探索。特别值得注意的是，模型在 RL 训练过程中展现出了代码自我修正的涌现能力。当生成的代码因函数未定义等错误而执行失败时，模型能够识别错误并自主生成修正后的可执行代码，这种元认知能力的出现标志着一个"顿悟时刻"（aha moment），表明模型已经掌握了自适应工具使用的能力。

训练过程中的行为演化分析揭示了几个关键趋势。首先，响应长度在 RL 训练后减少了约 40%（从 10k token 降至 6k token），这表明代码驱动的推理方法通过用简洁的代码替代复杂的计算过程，显著提升了推理 token 的利用效率。其次，代码比率、代码行数和正确代码数量在训练过程中持续上升，代码调用时机也逐渐提前，这些趋势共同表明模型的代码使用能力和策略性工具使用能力在不断发展。此外，代码用途分析显示，RL 训练后模型的代码目的变得更加多样化，从主要的计算和验证扩展到更广泛的问题类型，展现了自适应工具选择的元认知发展。

### 准备 RL 训练数据

强化学习阶段的核心训练数据是 BytedTsinghua-SIA/DAPO-Math-17k 数据集。这是一个大规模的数学问题集合，包含 179 万个数学问题及其答案，涵盖了从基础算术到高等数学的广泛主题。数据集的规模确保了模型能够在多样化的问题场景中学习和泛化工具使用策略。下载该数据集：

```bash
huggingface-cli download --repo-type dataset --resume-download \
    BytedTsinghua-SIA/DAPO-Math-17k \
    --local-dir /dataset/BytedTsinghua-SIA/DAPO-Math-17k
```

评估数据集使用 BytedTsinghua-SIA/AIME-2024，这是美国数学邀请赛（American Invitational Mathematics Examination）2024 年的试题集，包含 30 个高难度数学竞赛问题。AIME 是美国最具挑战性的高中数学竞赛之一，其题目需要深度推理和创造性的问题解决能力。该数据集应该在 SFT 阶段已经下载，位于 `/dataset/BytedTsinghua-SIA/AIME_2024` 目录。如果需要重新下载：

```bash
huggingface-cli download --repo-type dataset --resume-download \
    BytedTsinghua-SIA/AIME-2024 \
    --local-dir /dataset/BytedTsinghua-SIA/AIME_2024
```

值得注意的是，RL 训练数据集的规模远大于 SFT 阶段。这种设计是有意为之的：SFT 阶段通过少量高质量示例建立基础能力，而 RL 阶段则通过大规模数据的探索学习来提升模型的泛化能力和策略优化。DAPO-Math-17k 的 179 万问题为模型提供了充足的探索空间，使其能够在各种数学问题场景中发现有效的工具使用模式。

### 启动强化学习训练

在确保 SandboxFusion 服务正在运行的前提下，可以启动强化学习训练。首先确保已经切换回 verl 的 conda 环境，然后进入配方目录并执行 RL 训练脚本：

```bash
conda activate verl
bash recipe/retool/run_qwen2-32b_dapo.sh
```

强化学习训练使用 DAPO（Direct Alignment from Preferences Optimization）方法，这是一种高效的偏好优化算法。在训练过程中，模型会生成多个候选答案，并通过与代码沙箱的交互来验证答案的正确性。正确的答案会获得正向奖励，而错误的答案则会获得负向奖励。通过这种奖励机制，模型逐渐学会生成更准确的推理过程和更有效的工具使用策略。

强化学习阶段的训练时间通常比 SFT 阶段更长，因为每个训练步骤都涉及模型推理、代码执行和奖励计算等多个环节。训练过程同样会记录到 wandb 平台，可以实时监控奖励值、成功率等关键指标的变化趋势。随着训练的推进，应该能够观察到模型在数学问题求解上的准确率逐步提升，工具使用的效率也会不断优化。

### 训练监控与评估

在强化学习训练期间，需要密切关注几个关键指标来评估训练效果。首先是平均奖励值，它反映了模型生成答案的整体质量。随着训练的进行，平均奖励应该呈现上升趋势。其次是问题求解成功率，即模型能够正确回答的问题比例。这个指标直接反映了模型的实际能力提升。此外，还应关注工具调用频率和工具使用成功率，这些指标能够揭示模型是否学会了有效利用代码沙箱来辅助推理。

通过 wandb 界面，可以可视化这些指标的变化曲线，并与 SFT 阶段的基线进行对比。如果发现训练停滞或性能下降，可能需要调整学习率、奖励函数权重或其他超参数。定期在验证集上评估模型性能，有助于及时发现过拟合等问题并采取相应的调整措施。

### RL 训练过程实例

在 RL 训练过程中，可以观察到模型与 SandboxFusion 代码沙箱的实时交互。训练日志清晰地展示了模型生成代码、沙箱执行代码并返回结果的完整循环。以下是训练过程中 SandboxFusion 服务的典型日志输出：

```
2025-10-01 08:10:56 [debug] start processing python request with code ```
import math

x_approx = (4 * math.sqrt(3) - 2) / 5
print(f"Approximate x: {x_approx}")
``` and files []...(memory_limit: 1024MB) [sandbox.server.sandbox_api]
2025-10-01 08:10:56 [debug] running command python /tmp/tmppzrv67yh/tmp1y8y74j1.py [sandbox.runners.base]
2025-10-01 08:10:56 [debug] stop running command python /tmp/tmppzrv67yh/tmp1y8y74j1.py [sandbox.runners.base]

2025-10-01 08:10:57 [debug] start processing python request with code ```
from itertools import product

# Define all edges with indices (0-11)
edges = {
    'T1': 0, 'T2': 1
``` and files []...(memory_limit: 1024MB) [sandbox.server.sandbox_api]
2025-10-01 08:10:57 [debug] running command python /tmp/tmpy_ac6y5a/tmpxn02qp2v.py [sandbox.runners.base]
2025-10-01 08:10:57 [debug] stop running command python /tmp/tmpy_ac6y5a/tmpxn02qp2v.py [sandbox.runners.base]

2025-10-01 08:11:04 [debug] start processing python request with code ```
def is_greedy_successful(N):
    # Calculate the greedy result
    q = N // 25
    r = N % 25
    gr
``` and files []...(memory_limit: 1024MB) [sandbox.server.sandbox_api]
2025-10-01 08:11:04 [debug] running command python /tmp/tmpyqtl99_8/tmph_t_tj6u.py [sandbox.runners.base]
2025-10-01 08:11:04 [debug] stop running command python /tmp/tmpyqtl99_8/tmph_t_tj6u.py [sandbox.runners.base]

2025-10-01 08:11:07 [debug] start processing python request with code ```
import math

z_numerator = 9 * math.sqrt(5) - 1
z = z_numerator / 4
print(f"z = {z}")
``` and files []...(memory_limit: 1024MB) [sandbox.server.sandbox_api]
2025-10-01 08:11:07 [debug] running command python /tmp/tmpbk3a7frj/tmp12g7qyuf.py [sandbox.runners.base]
2025-10-01 08:11:07 [debug] stop running command python /tmp/tmpbk3a7frj/tmp12g7qyuf.py [sandbox.runners.base]
```

这些日志展示了 RL 训练的几个重要特征。首先，模型在处理不同类型的数学问题时会生成多样化的代码片段，从简单的数值计算（使用 math 模块）到复杂的算法实现（使用 itertools 等标准库）。其次，每个代码请求都在隔离的临时目录中执行，确保了执行环境的安全性和独立性。沙箱为每次执行创建独特的临时文件（如 `tmp1y8y74j1.py`），执行完成后立即清理，避免了状态污染。

从日志的时间戳可以看出，代码执行的速度非常快，大多数简单计算在毫秒级完成。这种高效的执行机制使得模型能够在训练过程中进行大量的工具调用尝试，快速积累经验。日志中还可以观察到，模型会针对同一个问题生成多个候选解决方案（体现在相近时间戳的多个代码请求），这正是 DAPO 算法基于 GRPO 框架的工作方式——通过生成和比较多个响应来估计相对优势。

值得注意的是，每个代码请求都设置了内存限制（memory_limit: 1024MB），这是沙箱环境的安全机制之一，防止恶意或低效的代码消耗过多系统资源。这种资源限制确保了训练过程的稳定性，即使模型生成了有问题的代码，也不会影响整个训练系统的运行。

### 训练启动与初始验证

当 RL 训练启动时，系统会输出详细的初始化和验证信息。首先是 AgentLoopWorker 的初始化日志，显示了代码解释器工具的配置：

```
(AgentLoopWorker pid=235550) Performing class-level ToolAgentLoop initialization [repeated 7x across cluster]
(AgentLoopWorker pid=235550) {
(AgentLoopWorker pid=235550)   "type": "function",
(AgentLoopWorker pid=235550)   "function": {
(AgentLoopWorker pid=235550)     "name": "code_interpreter",
(AgentLoopWorker pid=235550)     "description": "A tool for executing code.",
(AgentLoopWorker pid=235550)     "parameters": {
(AgentLoopWorker pid=235550)       "type": "object",
(AgentLoopWorker pid=235550)       "properties": {
(AgentLoopWorker pid=235550)         "code": {
(AgentLoopWorker pid=235550)           "type": "string",
(AgentLoopWorker pid=235550)           "description": "The code to execute."
(AgentLoopWorker pid=235550)         }
(AgentLoopWorker pid=235550)       },
(AgentLoopWorker pid=235550)       "required": ["code"]
(AgentLoopWorker pid=235550)     }
(AgentLoopWorker pid=235550)   }
(AgentLoopWorker pid=235550) }
(AgentLoopWorker pid=235550) Initialized tools: {'code_interpreter': <recipe.retool.retool.CustomSandboxFusionTool object at 0x7b4207c44c20>}
```

这些日志表明系统正在初始化 8 个 AgentLoopWorker（对应 8 张 GPU），每个 worker 都配置了相同的代码解释器工具。工具定义采用标准的函数调用格式，包含工具名称、描述和参数规范。"repeated 7x across cluster" 表示这个初始化过程在除主进程外的 7 个 worker 进程中重复执行，确保了分布式训练中各个进程的配置一致性。

接下来是初始验证阶段（val_before_train），这是 RL 训练前的基线性能评估。系统会在 AIME 2025 验证集上生成多个响应（30 个），并计算各种统计指标：

```
(TaskRunner pid=221183) Initial validation metrics: 
  'val-core/aime_2025/acc/mean@30': 0.1856 (平均准确率 18.56%)
  'val-core/aime_2025/acc/best@30/mean': 0.6362 (最佳答案准确率 63.62%)
  'val-core/aime_2025/acc/maj@30/mean': 0.2778 (多数投票准确率 27.78%)
  'val-aux/num_turns/min': 2 (最少交互轮数)
  'val-aux/num_turns/max': 16 (最多交互轮数)
  'val-aux/num_turns/mean': 6.59 (平均交互轮数)
```

这些指标揭示了训练前模型的几个重要特征。mean@30 表示对每个问题生成 30 个响应后的平均准确率，这个基线为 18.56%。best@30 表示在这 30 个响应中选择最佳答案的准确率，达到 63.62%，这说明模型有能力生成正确答案，但缺乏一致性。maj@30 表示通过多数投票选择答案的准确率为 27.78%，高于平均值但低于最佳值，说明多样化采样能够一定程度改善结果。

奖励值的统计同样重要。验证集上的平均奖励为 -0.464，这是因为奖励设计为正确答案 +1、错误答案 -1，负值表明初始模型的错误率较高。best@2、best@4、best@8 等指标分别表示在 2、4、8 个响应中选择最佳答案的平均奖励，这些值随着候选数量增加而提升（从 -0.263 到 0.290），验证了采样多样性对性能的积极影响。worst 指标则展示了最差答案的表现，始终保持负值且随候选数量增加而下降，这是预期行为。

num_turns 统计揭示了模型的交互模式。平均交互轮数为 6.59，范围从 2 到 16，说明模型在不同问题上采用了不同的策略，简单问题可能只需要少量交互，而复杂问题则需要多轮迭代。这个基线数据为后续观察训练过程中工具使用模式的演化提供了参照点。

训练进度条显示数据集完整遍历需要 3499 个训练步骤（基于 DAPO-Math-17k 数据集大小 179 万问题 / 批次大小 512 × 1 epoch）。但根据论文实验结果，ReTool 方法的一个显著优势是训练效率高，实际上只需要约 400 个训练步骤就能达到优异的性能。这意味着模型仅使用数据集的约 11% 就能充分学习工具使用策略，大幅缩短了训练周期。随着训练的进行，可以通过 wandb 界面实时监控这些指标的变化趋势，观察模型性能的提升轨迹。

### RL 训练循环机制

初始验证完成后，系统进入正式的 RL 训练循环。ReTool 采用基于 PPO 的训练流程，每个训练步骤都包含采样（Rollout）、奖励计算（Reward Computation）和策略更新（Policy Update）三个核心阶段，这个循环会在整个训练过程中反复执行。

**Rollout 阶段（采样与推理）**是训练循环中最耗时的部分。当前策略模型从训练集中抽取一批数学问题（每个 iteration 处理 32 个问题），对每个问题生成多个候选解答。根据配置参数 n_resp_per_prompt=16，系统会对每个问题生成 16 个不同的响应，总计 512 个响应。这些响应通过采样策略（temperature=1.0）产生，确保了足够的多样性以支持 DAPO 算法的相对优势估计。

在生成过程中，系统使用 vllm 推理引擎，配置了 infer_tp=4 的张量并行，将模型切分到 4 张 GPU 上进行高效推理。关键的是，模型与 SandboxFusion 代码沙箱进行实时交互。当模型生成代码片段并使用 `<code></code>` 标签标记时，vllm 暂停生成，将代码发送到沙箱执行。沙箱采用异步架构，维护一个包含 128 个 worker 的池，独立处理代码执行任务。执行完成后，结果（或错误信息）被包装在 `<interpreter></interpreter>` 标签中返回给模型。模型接收到反馈后继续生成，可以根据执行结果调整推理方向或尝试修正代码。这种交互可以重复多次，直到模型给出最终答案。每个响应都是一个完整的推理轨迹，包含交织的文本推理、代码片段和解释器输出。由于涉及代码执行等待和多轮交互，Rollout 阶段通常占据每个 iteration 的主要时间。

训练过程中可能出现工具调用解码错误，这是正常现象：

```
(AgentLoopWorker pid=235547) ERROR:2025-10-01 08:14:21,179:Failed to decode tool call: Invalid \escape: line 2 column 135 (char 135)
(AgentLoopWorker pid=235547) ERROR:2025-10-01 08:14:28,934:Failed to decode tool call: Extra data: line 2 column 228 (char 228)
(AgentLoopWorker pid=235548) ERROR:2025-10-01 08:15:37,582:Failed to decode tool call: Invalid \escape: line 2 column 480 (char 480)
```

这些错误表明模型在探索阶段生成了格式不完全正确的工具调用代码，比如包含未转义的反斜杠、多余的数据或缺少分隔符等。这正是强化学习探索过程的体现——模型需要尝试各种可能的代码生成方式，包括一些会失败的尝试，然后通过奖励信号学习哪些是有效的。随着训练的进行，这类错误的频率会逐渐降低，因为模型学会了生成格式正确的工具调用。

**奖励计算阶段（Reward Computation）**是一个相对快速的 CPU 密集型过程。系统收集 Rollout 阶段生成的全部 512 个响应，提取每个响应的最终答案（通常包含在 `<answer></answer>` 或 `\boxed{}` 标记中）。答案提取器会处理各种格式变体，确保能够可靠地识别模型给出的数值、表达式或符号答案。然后将提取的答案与标准答案进行比较，使用等价性检查而非简单的字符串匹配，例如数学表达式 "1/2" 和 "0.5" 会被判定为等价。正确答案获得 +1 奖励，错误答案获得 -1 奖励。

这种简单的二值奖励设计是 ReTool 方法的一个重要特点。与一些方法尝试对代码可执行性、中间步骤正确性等给予额外奖励不同，ReTool 只关注最终结果，让模型完全基于结果反馈来学习。这种设计避免了复杂的启发式规则可能引入的偏差，给予模型更大的探索自由度。对于每个问题的 16 个响应，系统计算出 16 个对应的奖励值。这些奖励值在 DAPO 算法中用于估计相对优势——每个响应的质量相对于同一问题的其他响应如何。同一问题的多个响应之间的奖励差异直接反映了不同策略的优劣，为策略梯度提供了明确的学习信号。DAPO 的 Token-Level Policy Gradient Loss 确保这些奖励信号被公平地分配到不同长度的响应上，避免超长响应主导训练过程。

**策略更新阶段（Policy Update）**是 PPO 算法的核心，也是唯一需要反向传播的阶段。系统将 train_batch_size=512 个样本（来自 32 个问题 × 16 个响应）作为一个完整的训练批次。为了适应 GPU 显存限制并提高训练稳定性，这个大批次被进一步分割成若干个 ppo_mini_batch_size=64 的小批次，依次进行梯度下降。具体来说，512 个样本被分成 8 个 mini-batch，每个包含 64 个样本。

对于每个 mini-batch，训练过程如下：首先，策略模型对这 64 个响应进行前向传播，计算在当前策略下生成这些响应的对数概率。然后，计算策略比率，即当前策略相对于旧策略（生成这些响应时的策略）生成该响应的概率比。接下来，使用 DAPO 算法估计的优势函数和改进的裁剪目标函数计算损失。裁剪机制通过 clip_ratio_low=0.2 和 clip_ratio_high=0.28 实现不对称裁剪，这是 DAPO 的 Clip-Higher 策略的核心——为向上更新提供 28% 的空间，为向下更新仅提供 20% 的空间，确保新策略在探索高奖励模式的同时不会出现性能退化。Token-Level Policy Gradient Loss 在此阶段发挥作用，对损失函数按响应长度进行归一化，确保长短序列获得公平的梯度权重。最后，进行反向传播和参数更新。

由于启用了 FSDP 和 CPU offload，这个阶段涉及复杂的内存管理。模型参数和优化器状态在 CPU 和 GPU 之间频繁传输。每个 mini-batch 的训练开始时，相关参数从 CPU 加载到 GPU；计算完成后，更新的参数和梯度信息又被传回 CPU。全部 8 个 mini-batch 处理完成后，一次完整的策略更新就完成了。DAPO 继承了 GRPO 不需要单独训练价值网络的优势，通过同批次响应的相对质量来估计优势函数，大幅简化了训练流程并减少了显存需求。在此基础上，DAPO 的四项技术创新进一步提升了训练的效率和稳定性，使其能够在 50% 的训练步数内超越传统方法的性能。

这个采样-奖励-更新的循环在整个训练过程中持续进行。每完成一个训练步骤，策略模型的参数就得到一次更新，生成能力略有提升。然后进入下一个循环，使用更新后的策略重新采样，获得新的响应和奖励，再次更新策略。正是通过这种迭代优化，模型逐渐学会了何时调用工具、如何编写有效代码、怎样利用执行结果来辅助推理等复杂技能。

根据配置，每 5 个训练步骤（test_freq=5）会进行一次验证，评估当前策略在验证集上的表现。每 30 个步骤（save_freq=30）会保存一次模型检查点，用于后续恢复或分析。根据论文实验结果，训练约 400 个这样的循环就能达到 67% 的 AIME 2024 准确率，显著超越仅用文本推理的基线模型（40%）。这种高效的训练特性使得 ReTool 在实际应用中具有很强的可行性。

随着训练的深入，可以观察到几个重要的演化趋势。代码使用频率逐渐提高，模型学会在更多问题上主动调用工具。代码的复杂度增加，从简单的计算扩展到复杂的算法实现。工具调用的时机提前，模型学会尽早引入代码来避免累积误差。最引人注目的是涌现行为的出现——模型开始展现代码自我修正能力，能够从执行错误中学习并生成修正版本，这种元认知能力标志着模型已经掌握了真正自适应的工具使用策略。

### Rollout 时间估算方法

理解 Rollout 阶段的时间构成对于合理规划训练周期至关重要。虽然 SandboxFusion 代码沙箱能够高速处理代码执行请求（每秒处理数个请求），但整个 Rollout 阶段的耗时远不止代码执行本身。

**实际工作量计算**涉及多个维度的乘积。根据配置参数，每个训练步骤需要生成 train_batch_size=512 个完整响应，这实际上是从 32 个数学问题（512 / n_resp_per_prompt = 512 / 16 = 32）分别生成 16 个不同的候选答案。每个响应不是一次性生成，而是通过多轮交互逐步构建的推理轨迹。从实际训练数据可以看出，平均每个响应包含约 7.82 轮交互（num_turns/mean），每轮交互通常包含一次文本推理生成、一个代码片段的生成和执行、以及解释器反馈的处理。因此，完成一个完整的 Rollout 需要的代码执行次数为：512 个响应 × 7.82 轮/响应 ≈ 4000 次代码执行。这个数字远超表面上的 512 个响应，解释了为什么即使 SandboxFusion 每秒能处理多个请求，整个 Rollout 阶段仍需要较长时间。

**Token 生成量和实际性能表现**可以从训练日志中获得准确数据。从第一个训练步骤的统计来看，512 个响应的平均长度为 2707 tokens，范围从最短的 288 tokens 到达到上限的 16384 tokens。约 0.85% 的响应（4-5 个）达到了最大长度限制被截断。整个训练步骤处理的总 token 数为 24,892,834 tokens，这包括了 Rollout 生成、Reference Model 的 log probability 计算以及 Actor 更新过程中的前向和反向传播。系统的整体吞吐量为 641 tokens/秒，显著低于纯文本生成场景，这是多轮交互、代码执行等待和 CPU offload 等因素共同作用的结果。

**训练各阶段的实际耗时**揭示了计算资源的分配模式。从第一个训练步骤的统计数据看，Rollout 生成阶段耗时 2529 秒（42 分钟），这是训练循环中最耗时的阶段，期间 vllm 引擎需要生成 512 个响应并与 SandboxFusion 进行约 4000 次代码执行交互。Reference Model 的 log probability 计算耗时 525 秒（8.7 分钟），这个阶段对已生成的 512 个响应进行前向传播以获取旧策略下的概率值，相比 Rollout 阶段要快得多。策略更新阶段耗时 1795 秒（30 分钟），这个阶段将 512 个样本分成 8 个 mini-batch 依次进行梯度下降，由于启用了 FSDP 和 CPU offload，大量时间消耗在参数的 CPU-GPU 传输上，因此成为除 Rollout 外的第二大时间开销。整个训练步骤的总耗时为 4853 秒（约 81 分钟），这个时间**不包括**初始验证阶段，初始验证在训练正式开始前独立完成。

**批处理效率下降**是多轮交互场景的固有挑战。虽然 vllm 配置了异步模式（mode=async），但在实际运行中，512 个请求的进度并不同步。开始时，vllm 可能以接近满批次的状态处理请求，但随着部分请求遇到代码执行触发器并暂停等待，实际的并发批次大小会逐渐下降。当一些请求在等待沙箱返回结果时，其他请求继续生成；当暂停的请求收到反馈后重新加入批次时，又有新的请求遇到代码触发器而暂停。这种动态的批次大小波动导致实际的平均批处理大小远小于理论上的 512，进而降低了整体吞吐量。

**通过 SandboxFusion 日志估算进度**是一个实用的监控方法。通过统计日志条目的数量，可以粗略估算已完成的响应数。基于实际统计，平均每个响应约有 8 次代码调用（num_turns/mean: 7.82），因此已完成响应数 ≈ 代码执行请求数 / 8。进度百分比 = 已完成响应数 / 512 × 100%。

**第一个训练步骤的实际表现**提供了宝贵的性能基准。根据实际运行数据，完整的第一个训练步骤耗时 4853 秒（约 81 分钟），处理了总计 24,892,834 个 tokens，整体吞吐量为 641 tokens/秒。训练过程的平均奖励从初始验证的 -0.464 提升到 0.058，表明模型在单个训练步骤后就开始展现学习效果。响应长度统计显示，平均响应为 2707 tokens，最短 288 tokens，最长达到上限的 16384 tokens。约 0.85% 的响应（4-5 个）达到了最大长度限制被截断。输入提示的平均长度为 332 tokens，最长 787 tokens，说明数学问题的描述相对简洁，而模型的推理过程则相当详细。

**资源利用和性能指标**反映了训练系统的效率水平。GPU 显存方面，单步训练的峰值显存分配达到 214.8 GB，显存预留为 226.8 GB，接近 8 张 H200 GPU 的总显存容量。CPU 内存使用达到 213.0 GB，这是启用 CPU offload 后将大量模型参数和优化器状态卸载到 CPU 内存的结果。模型的 FLOPs 利用率（MFU）为 39.7%，这个数值虽然低于纯计算场景的理论峰值，但考虑到多轮交互的等待、CPU-GPU 数据传输以及异步 rollout 的调度开销，是一个合理的利用水平。最慢的单个响应生成耗时 2492 秒（41.5 分钟），该响应达到了最大长度 16384 tokens，其生成时间几乎决定了整个 Rollout 阶段的总耗时，体现了典型的"木桶效应"。

**后续训练步骤的稳定性与学习曲线**可以从 step 2 的数据中得到验证。第二个训练步骤的总耗时为 4859 秒（81 分钟），与第一步几乎完全一致，表明训练流程已经进入稳定状态，初始化开销并非主要瓶颈。各阶段的时间分布保持高度一致：Rollout 生成 2543 秒（42.4 分钟），log probability 计算 510 秒（8.5 分钟），策略更新 1801 秒（30 分钟）。然而，在训练效果方面，模型展现出了快速的学习进展。平均奖励从 step 1 的 0.058 大幅跃升至 step 2 的 0.164，提升了 182%，对应的准确率从约 53% 提升到 58%。这种显著的性能改善验证了 ReTool 方法的有效性——模型正在快速学习何时以及如何使用代码工具来提升数学推理能力。

值得注意的是，响应模式出现了分化。最短响应从 288 tokens 降至 60 tokens，表明模型学会了对简单问题采用更直接的解法；而最长响应仍然达到 16384 tokens 上限，且被截断的响应比例从 0.85% 增加到 0.94%，说明对复杂问题的探索更加深入。平均交互轮数略微下降（从 7.82 到 7.79），但结合奖励提升来看，这反映了工具使用效率的改善而非探索深度的减少。系统的整体吞吐量从 641 tokens/s 提升到 650 tokens/s，虽然提升幅度不大，但表明训练流程在持续优化。CPU 内存使用从 213 GB 增加到 227 GB，这种增长趋势需要持续监控，以防在长时间训练后出现内存溢出。

**前十个训练步骤的实际观察**揭示了训练过程的动态特征和性能演化趋势。步骤一和步骤二的耗时高度一致，分别为 4853 秒和 4859 秒（均为 81 分钟），表明训练流程在初始阶段已经进入稳定状态。然而，从步骤三开始出现了显著的耗时增长现象，这一趋势持续到步骤九，总耗时从 81 分钟增长到 136 分钟，增幅达 68%。步骤十的总耗时为 128 分钟（包含 19.4 分钟的验证时间），纯训练时间为 109 分钟，虽然比步骤九略有下降，但仍然显著高于初始水平。

这种耗时增长的主要驱动因素是响应长度的持续增加。平均响应长度从步骤一的 2707 tokens 持续增长至步骤九的 4330 tokens，累计增长达 60%。步骤十首次出现了响应长度的回落，降至 4116 tokens，下降了 5%，这可能标志着响应长度增长趋势开始趋于稳定。被截断响应的比例同样经历了大幅增长，从步骤一的 0.85%（4-5 个响应）增长到步骤九的 3.70%（19 个响应），步骤十略有回落至 2.93%（15 个响应）。这些触及 16384 tokens 上限的超长响应，其生成时间从步骤一的 41.5 分钟增长到步骤九的 76.2 分钟，步骤十回落至 70.8 分钟，始终主导着整个 Rollout 阶段的总耗时。

步骤五是第一个包含验证的步骤，总耗时达到 6069 秒（101.2 分钟），其中验证阶段单独耗时 913 秒（15.2 分钟）。验证阶段需要对 30 个问题分别生成 30 个响应（n_resp_per_prompt_val=30），总计 900 个响应。有趣的是，验证阶段的平均交互轮数为 6.98，明显低于训练阶段的 7.84，这可能与验证时使用的不同采样参数（top_p=0.6, temperature=1.0）有关。扣除验证时间后，步骤五的纯训练时间为 5156 秒（86 分钟），仍然高于前两步的 81 分钟，但低于步骤四的 94 分钟，显示出一定的波动性。

**模型学习进展**在前十个步骤中呈现出波动上升的轨迹。训练集上的平均奖励从步骤一的 0.058 波动上升至步骤十的 0.251，对应的准确率从约 53% 提升到 63%，累计提升了 10 个百分点。学习曲线呈现非单调特征，步骤一到二有大幅提升（+5 个百分点），步骤二到五增速放缓，步骤六达到局部峰值（0.251，63%），步骤七出现回落（0.206，60%），随后在步骤八再次攀升（0.264，63%），步骤九和十维持在 0.25 左右的水平。这种波动是强化学习训练的正常现象，反映了策略在探索和优化之间的动态平衡。

验证集性能的演化提供了模型泛化能力的关键洞察。步骤五的验证结果显示，平均准确率（mean@30）为 27.9%，Best-of-30 指标为 62.9%，多数投票准确率（maj@30）为 36.9%。到步骤十，平均准确率略微提升至 28.3%，但 Best-of-30 指标显著提升至 67.4%，增长了 4.5 个百分点，表明模型的峰值能力在持续增强。然而，多数投票准确率下降至 33.8%，降低了 3.1 个百分点，这种分化现象说明模型虽然能够生成更高质量的答案，但一致性有所下降。验证阶段的平均交互轮数从步骤五的 6.98 降至步骤十的 5.85，减少了 16%，这与训练阶段的交互模式演化（轮数减少但每轮更复杂）呈现相似的趋势。训练集与验证集之间存在明显的性能差距，训练集准确率达到 63%，而验证集平均准确率仅为 28%，但 Best-of-30 达到 67.4%，说明模型具备生成正确答案的能力，主要挑战在于提高生成的一致性和可靠性。

**响应生成模式的演化**在前十个步骤中展现出先增长后稳定的趋势。平均响应长度从步骤一的 2707 tokens 持续增长至步骤九的 4330 tokens，累计增长达 60%，步骤十首次回落至 4116 tokens。平均交互轮数则呈现相反的趋势，从步骤一的 7.82 轮逐渐降至步骤十的 7.10 轮，下降了 9%，但每轮交互产生的内容大幅增加，从平均每轮 346 tokens 增长到步骤十的 580 tokens，增幅达 68%。这种模式转变表明模型正在学习生成更复杂、更详细的单轮推理，而不是简单地增加交互轮数。最慢的单个响应生成时间从步骤一的 41.5 分钟增长到步骤九的 76.2 分钟，步骤十回落至 70.8 分钟，这些超长响应的生成时间几乎完全决定了整个 Rollout 阶段的总耗时，木桶效应贯穿整个训练过程。整个训练步骤处理的 token 总量从步骤一的 24.9M 增长到步骤十的 36.5M，增幅达 47%，这种增长既来自响应长度的增加，也来自训练流程中多次前向传播的累积效应。

**系统资源利用情况**在前十个步骤中展现出良好的稳定性。GPU 显存使用从步骤一的 214.8 GB 逐渐增长到步骤十的 216.1 GB，增长不到 1 GB，始终保持在 8 张 H200 GPU 总容量的安全范围内。CPU 内存使用在步骤二达到 227 GB 后在 220-228 GB 范围内波动，步骤十为 221.8 GB，没有出现持续增长导致内存溢出的风险。模型的算力利用率（MFU）在 39.7%-40.3% 之间小幅波动，保持在稳定的水平，这个利用率虽然低于纯计算场景的理论峰值，但考虑到多轮交互等待、CPU-GPU 数据传输和异步调度等开销，反映了当前配置下的实际表现。整体吞吐量从步骤二的 650 tokens/s 峰值持续下降到步骤十的 595 tokens/s，下降了 8.5%，这种下降与响应长度增加和处理更长序列的额外开销直接相关。

**训练内部技术指标**展现了 DAPO 算法的稳定运行状态和设计优势。策略梯度损失（actor/pg_loss）在前十个步骤中保持在 -0.0022 到 +0.0011 之间的小幅波动，绝对值很小且稳定，表明策略更新处于平稳状态。策略裁剪比例（actor/pg_clipfrac）从步骤一的 0.197% 逐渐下降到步骤十的 0.157%，始终保持在很低的水平，说明几乎所有的策略更新都在允许的范围内，没有触发过度更新的保护机制。这个持续的低裁剪比例验证了 Clip-Higher 策略的有效性——通过不对称裁剪（clip_ratio_low=0.2, clip_ratio_high=0.28），为向上更新预留了足够空间，避免了过度裁剪导致的训练停滞。下界裁剪比例（actor/pg_clipfrac_lower）在所有步骤中都接近于零（最大仅为 1.2e-7），表明没有出现策略退化的情况，证明了严格限制向下更新的策略是成功的。

KL 散度（actor/ppo_kl）在前十个步骤中始终保持在 1.3e-5 到 2.4e-5 之间（除步骤五的异常低值 1.51e-6），数值极小且波动范围很窄。步骤十的 KL 散度为 1.46e-5，处于正常范围内。这种极小的 KL 散度配合较低的裁剪比例，共同保证了训练的稳定性，避免了策略崩溃或性能突然下降的风险。由于配置中设置了 kl_coef=0.0，KL 散度项并未直接加入损失函数，这些观察到的小 KL 值完全是裁剪机制和小学习率（1e-6）自然产生的结果，体现了 DAPO 算法内在的稳定性。

梯度范数（actor/grad_norm）在前十个步骤中呈现持续的下降趋势，从步骤一的 0.132 降至步骤十的 0.104，累计下降了 21%。梯度范数的大小直接反映了参数更新的强度，其下降趋势说明模型参数在逐渐接近某个局部最优点，需要的调整幅度越来越小。所有步骤的梯度范数都保持在 0.096-0.133 的健康范围内，既没有出现梯度消失也没有梯度爆炸，表明反向传播过程运行良好。特别值得关注的是策略熵（actor/entropy）的演化轨迹，从步骤一的 0.158 逐渐上升至步骤十的 0.180，增长了 14%。这种熵值的持续上升是 DAPO 算法 Clip-Higher 策略成功的有力证据——在使用对称裁剪的传统 GRPO 基线中，熵往往会在训练早期快速下降到接近零，导致模型失去探索能力和输出多样性。DAPO 通过不对称裁剪成功维持并增强了策略的探索性，避免了熵崩溃问题。

优势函数（critic/advantages）的统计分布在各步骤中保持相对一致的模式。平均优势值在 0.013-0.054 之间波动，接近于零，这符合 DAPO 继承自 GRPO 的设计——优势函数衡量的是相对于批次平均水平的好坏，因此其均值理论上应该接近零。优势函数的极值范围稳定在 -3.75 到 +3.75 之间，这个对称的范围表明批次中既有显著优于平均水平的优秀响应，也有明显劣于平均水平的较差响应，为策略梯度提供了清晰的学习信号。Token-Level Policy Gradient Loss 的应用确保了这些优势值被公平地分配到不同长度的序列上，避免了长序列因为包含更多 token 而获得不成比例的大梯度。优势函数分布的稳定性表明算法的相对质量估计机制运行正常，能够有效识别不同响应的优劣。

### RL 训练配方参数详解

强化学习训练脚本的参数配置比 SFT 阶段更加复杂，涉及到策略优化、奖励计算和多轮交互等多个维度。理解这些参数对于充分发挥 ReTool 方法的潜力至关重要。

数据和模型路径配置定义了训练的基础资源。train_files 指向 DAPO-Math-17k 数据集，这是包含 179 万数学问题的大规模训练语料。test_files 在这个配置中指向 aime_2025，用于训练过程中的定期评估。model_path 参数指向 SFT 阶段训练完成并转换为 Hugging Face 格式的模型检查点，这个模型已经具备了基本的工具使用能力，RL 训练将在此基础上进一步优化。tool_config_path 指定 SandboxFusion 工具的配置文件路径，定义了代码执行环境的各项参数。

算法相关参数控制了强化学习的核心机制。adv_estimator 设置为 grpo（Group Relative Policy Optimization），这是 DAPO 算法的基础组件，负责通过相对优势估计来优化策略。DAPO 在 GRPO 的框架上增加了四项关键改进（Clip-Higher、Dynamic Sampling、Token-Level Loss 和 Overlong Reward Shaping），使其特别适合长思维链场景。与传统的 PPO 算法相比，GRPO 不需要训练单独的 critic 模型，而是通过同一批次中不同响应的相对质量来估计优势，大幅简化了训练流程并提高了样本效率。use_kl_in_reward 和 kl_coef 参数控制是否在奖励中加入 KL 散度惩罚项，这里都设置为 False 和 0.0，表示不使用显式的 KL 约束，完全依靠 DAPO 的裁剪机制来控制策略更新幅度。clip_ratio_low 和 clip_ratio_high 分别设置为 0.2 和 0.28，这个不对称的裁剪范围正是 DAPO 的 Clip-Higher 策略的体现——为向上的策略更新提供更大空间（0.28 > 0.2），鼓励探索高奖励模式的同时防止性能退化。

序列生成和批次配置参数决定了模型的推理方式。max_turns 设置为 8，表示允许模型最多进行 8 轮思考和工具调用的交互循环。这个设置为复杂问题的深度探索提供了充足空间，模型可以反复验证假设、调整策略。max_prompt_length 设置为 2048，max_response_length 设置为 16384，两者的差异反映了 ReTool 方法的特点：输入问题通常较短，但包含多轮交互和代码执行结果的响应序列可能非常长。train_batch_size 设置为 512 是一个相当大的批次，但实际上通过 n_resp_per_prompt=16 实现，即对每个问题生成 16 个不同的响应，然后在这些响应之间进行相对质量比较。ppo_mini_batch_size 设置为 64，表示在策略更新时将 512 个样本分成多个小批次进行梯度下降。

性能优化参数对于高效利用 GPU 资源至关重要。infer_tp 设置为 4，表示在推理阶段使用 vllm 引擎的张量并行，将模型切分到 4 张 GPU 上进行快速生成。train_sp 设置为 8，表示在训练阶段使用 Ulysses 序列并行，将长序列切分到全部 8 张 GPU 上处理。offload 设置为 True，启用参数和优化器状态的 CPU 卸载，这对于在有限显存下训练 32B 参数模型是必需的。虽然 CPU 卸载会带来一定的速度开销，但它使得在单台服务器上完成整个训练流程成为可能。

多轮交互配置体现了 ReTool 的核心特性。multi_turn.enable 启用多轮对话模式，max_user_turns 和 max_assistant_turns 都设置为 8，与 max_turns 参数对应。tool_config_path 指向 SandboxFusion 的配置文件，定义了工具调用的接口和执行环境。format 设置为 hermes，这是一种标准化的工具调用格式，确保模型生成的工具调用请求能够被 SandboxFusion 正确解析和执行。rollout 模式设置为 async，允许多个推理请求并发执行，充分利用 vllm 引擎的高吞吐能力。

采样策略参数影响响应的多样性。训练阶段通过 n_resp_per_prompt=16 对每个问题生成 16 个不同的响应，这些响应的多样性为 DAPO 算法的相对优势估计提供了丰富的比较基础。虽然当前配置使用固定的采样数量，但 DAPO 的 Dynamic Sampling 技术允许在训练过程中根据需要动态调整这个参数。验证阶段则通过 n_resp_per_prompt_val=30 生成更多响应，并使用 top_p=0.6 和 temperature=1.0 的采样参数，在保持一定随机性的同时避免生成过于离谱的答案。gpu_memory_utilization 设置为 0.9，表示 vllm 引擎可以使用 GPU 显存的 90%，为推理预留充足的 KV cache 空间。

训练控制参数决定了训练的节奏和监控方式。val_before_train 设置为 True，在训练开始前先进行一次验证，建立性能基线。log_val_generations 设置为 100，表示在验证时记录前 100 个生成样本到 wandb，便于人工检查模型的推理过程和工具使用情况。save_freq 设置为 30，每 30 个训练步骤保存一次检查点，test_freq 设置为 5，每 5 个步骤进行一次验证。total_epochs 设置为 1，因为 DAPO-Math-17k 数据集规模已经足够大，单次遍历即可获得显著的性能提升。actor_lr 设置为 1e-6，这是一个相对较小的学习率，因为模型已经在 SFT 阶段学会了基本能力，RL 阶段只需要进行精细调整。

## 参考资料

- [DAPO 论文：大规模 LLM 强化学习系统](https://arxiv.org/pdf/2503.14476)
- [ReTool 论文：多轮对话与代码沙箱提升数学推理](https://arxiv.org/pdf/2504.11536)
- [verl 官方仓库](https://github.com/volcengine/verl/)
- [SandboxFusion 官方仓库](https://github.com/bojieli/SandboxFusion)
- [Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [AIME 2024 数据集](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024)
- [DAPO-Math-17k 数据集](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

## 常见问题

### GPU 显存不足怎么办？

如果遇到 GPU 显存不足的问题，可以尝试以下几种方案：

1. 减小微批次大小（micro_batch_size_per_gpu）
2. 启用 CPU offload 选项
3. 减小序列长度（max_length）
4. 使用梯度累积来模拟更大的批次

### 训练中断如何恢复？

verl 框架支持从检查点恢复训练。在配置文件中设置 `resume_from_path` 参数指向之前保存的检查点目录即可。训练脚本会自动加载模型权重、优化器状态和训练进度。

### 如何评估模型性能？

模型训练完成后，可以在 AIME 测试集上进行评估。使用 verl 提供的评估脚本，输入测试数据和模型检查点路径，脚本会自动运行推理并计算准确率等指标。

### 多机训练如何配置？

对于多机训练，需要在每台机器上配置相同的环境，并在训练脚本中设置正确的节点数量（nnodes）和节点排名。verl 使用 PyTorch 的分布式训练机制，需要配置主节点的地址和端口供各节点通信。

