# Prompt Distillation with Hugging Face TRL

This project demonstrates **prompt distillation** - a technique to distill knowledge from a **thinking model with long prompts** into a **non-thinking model without prompts**, making responses dramatically faster.

## 🎯 Main Goal

Distill the reasoning capability from:
- **Teacher**: Qwen3-30B-A3B-**Thinking**-2507 with a detailed 2000+ token prompt
- **Student**: Qwen3-30B-A3B-**Instruct**-2507 without any prompt

**Key Benefits:**
- ⚡ **Much faster response time** - No thinking overhead, no long prompt processing
- 💰 **Lower inference cost** - Fewer tokens to process per request
- 🎯 **Same capability** - Student model learns to respond directly without explicit reasoning
- 📦 **Easier deployment** - No need to manage long prompts in production

## What is Prompt Distillation?

Prompt Distillation (also known as **context distillation**) is a training method that makes an LLM internalize a long and complex prompt into its parameters. In this experiment, we also remove the thinking overhead by distilling from a thinking model to a non-thinking model.

**Example - Language Classification:**

We want to internalize this detailed prompt:
> "Classify the language of the provided text into these labels: ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot. Use these rules: Devanagari script → hi, Greek script → el, Cyrillic script → ru..." *(2000+ tokens)*

**Before distillation (Teacher with thinking + prompt):**
```
System: <2000+ token detailed prompt>
User: 一生、バンドしてくれる？
Assistant: <thinking>Let me analyze the script... These are Han characters... Based on rule X...</thinking>ja
⏱️  Response time: ~2-3 seconds
```

**After distillation (Student, no thinking, no prompt):**
```
User: 一生、バンドしてくれる？
Assistant: ja
⏱️  Response time: ~0.1 seconds (20-30x faster!)
```

## Methodology

The method involves two stages:

1. **Data Generation (Teacher Model)**: A **thinking model** uses a detailed prompt to generate responses with explicit reasoning.
   - Teacher generates: `response = thinking_model(long_prompt, query)`
   
2. **Student Training (Distillation)**: A **non-thinking model** is fine-tuned to predict responses directly without the prompt or thinking process.
   - Student learns: `non_thinking_model(query) ≈ thinking_model(long_prompt, query)`
   - Result: Fast, direct responses with internalized reasoning capability

## Hyperparameters

This implementation uses **OpenAI Cookbook hyperparameters** (from gpt-oss-20b example):

| Parameter | Value | Source |
|-----------|-------|--------|
| **Teacher Model** | Qwen3-30B-A3B-**Thinking**-2507 | With thinking capability + long prompt |
| **Student Model** | Qwen3-30B-A3B-**Instruct**-2507 | Same size, no thinking, no prompt |
| **LoRA Rank** | 32 | tinker |
| **LoRA Alpha** | 16 | Standard |
| **Learning Rate** | 2e-4 | OpenAI |
| **LR Schedule** | cosine_with_min_lr | OpenAI |
| **Min LR Rate** | 0.1 | OpenAI |
| **Batch Size** | 4 per GPU | OpenAI |
| **Gradient Accumulation** | 4 steps | OpenAI |
| **Max Length** | 2048 | OpenAI (student only needs short context) |
| **Num Epochs** | 1 | OpenAI |
| **Temperature** | 0.15 | tinker (data generation) |
| **Warmup Ratio** | 0.03 | OpenAI |
| **Gradient Checkpointing** | True | OpenAI |

**Key Design Choice**: We use the same 30B model for both teacher and student. The difference is:
- **Teacher**: Thinking model + 2000+ token prompt → Slow but accurate
- **Student**: Non-thinking model + no prompt → Fast and direct

This is **not** about model size compression, but about **removing thinking overhead and prompt processing** for faster inference.

## Dataset

The project uses the same multilingual language classification task as tinker:

- **Task**: Classify text into 13 language labels
- **Labels**: `ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot`
- **Source Data**: `example-data/multilingual.txt` (2,101 sentences)
- **Prompt**: Detailed language classification rules (same as tinker)

## Installation

### Prerequisites

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Setup Weights & Biases for training monitoring:

```bash
# Login to wandb (required for training progress tracking)
wandb login

# Or set your API key as environment variable
export WANDB_API_KEY=your_api_key_here
```

You can get your API key from [https://wandb.ai/settings](https://wandb.ai/settings)

### System Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU acceleration)
- **GPU**: H100 80GB (for 30B model) or any GPU with 24GB+ for smaller models
- **Memory**: ~70-75GB VRAM for 30B model with LoRA

## Usage

### Step 1: Generate Training Data

Generate prompt distillation data using the teacher model:

```bash
# Single instance (uses tensor parallelism across GPUs)
python create_data.py \
    --input_file ./example-data/multilingual.txt \
    --output_file ./data/prompt_distillation_lang.jsonl \
    --model_name Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --temperature 0.15 \
    --tensor_parallel_size 4

# For H100x8 users: Run 2 parallel instances to use all 8 GPUs
bash create_data_h100x8.sh
```

**Options:**
- `--input_file`: Path to input sentences (one per line)
- `--output_file`: Where to save generated training data
- `--model_name`: Teacher model (Qwen3-30B-A3B-Thinking-2507 for better accuracy)
- `--temperature`: Sampling temperature (0.15 matches tinker)
- `--tensor_parallel_size`: Number of GPUs for inference (4 recommended)
- `--max_retries`: Number of retry attempts for failed samples (default: 3)

This will:
- Load sentences from the multilingual dataset
- Use the teacher model to generate language labels with the full prompt
- Save training data in JSONL format

**Output format:**
```json
{
  "messages": [
    {"role": "user", "content": "Text in some language"},
    {"role": "assistant", "content": "en"}
  ]
}
```

### Step 2: Train the Student Model

Fine-tune the student model on the distilled data using TRL:

```bash
# Single GPU training (recommended - simpler and works reliably)
bash train_trl.sh
```

**Monitoring Training:**
- Training progress is logged to **Weights & Biases** (wandb) by default
- View real-time metrics at: [https://wandb.ai](https://wandb.ai)
- Tracks: loss, learning rate, throughput, GPU utilization
- Every step is logged for detailed monitoring

**To disable wandb logging:**
```bash
python train_sft_trl.py --report_to none ...other args...
```

### Step 3: Evaluate Your Model

After training, evaluate the distilled model's performance:

```bash
# Evaluate with defaults (uses all defaults)
python evaluate.py

# Quick evaluation on a subset
python evaluate.py --max_samples 100

# Save results to a file
python evaluate.py --output_file ./evaluation_results.json

# Custom model path
python evaluate.py --model_path ./models/my_custom_model
```

**Defaults:**
- Model: `./models/prompt_distillation_trl`
- Base model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Test file: `./example-data/multilingual.txt`

**Real-time Output Example:**
```
Evaluating model...
================================================================================
✓ [   1/2095] Pred: ar | GT: ar | Acc: 1/1 (100.0%) | وقال، ماما، لقد عدت للمنزل.
✓ [   2/2095] Pred: en | GT: en | Acc: 2/2 (100.0%) | Hello, how are you today?
✗ [   3/2095] Pred: es | GT: fr | Acc: 2/3 ( 66.7%) | Bonjour, comment allez-vous?
✓ [   4/2095] Pred: zh | GT: zh | Acc: 3/4 ( 75.0%) | 你好，今天天气怎么样？
...
================================================================================
Evaluation completed: 2095 samples processed
```

Each line shows: prediction, ground truth, running accuracy, and the input text.

## Project Structure

```
prompt-distillation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── create_data.py                     # Data generation script (Step 1)
├── create_data_h100x8.sh              # Parallel data generation for H100x8
├── train_sft_trl.py                   # Training script using TRL (Step 2)
├── train_trl.sh                       # Training script (single GPU)
├── evaluate.py                        # Evaluation script
├── data/                              # Generated training data
│   └── prompt_distillation_lang.jsonl
└── models/                            # Trained model checkpoints
    └── prompt_distillation_trl/
```

## Why This Approach?

### Thinking Model → Non-Thinking Model

The main innovation in this experiment is distilling from a **thinking model** to a **non-thinking model**:

1. **Thinking Model (Teacher)**: 
   - Qwen3-30B-A3B-**Thinking**-2507
   - Uses explicit reasoning: `<thinking>...</thinking>`
   - Requires long prompts with detailed instructions
   - Slower but more accurate
   
2. **Non-Thinking Model (Student)**:
   - Qwen3-30B-A3B-**Instruct**-2507
   - No thinking tags, direct responses
   - No prompts needed in production
   - **20-30x faster inference**

### Why TRL Instead of verl?

We use Hugging Face TRL for this implementation because:

1. **More Common**: TRL is widely adopted in the community
2. **Better Documentation**: Extensive docs and examples
3. **Simpler Setup**: No need to convert JSONL to Parquet
4. **Standard Workflow**: Works seamlessly with HuggingFace ecosystem
5. **Easier to Debug**: Clear error messages and better tooling

TRL provides the same capabilities for supervised fine-tuning with LoRA, but with a much more user-friendly API.

## Key Implementation Details

### Data Format

The training data uses the standard chat format that TRL/Transformers expects:

```json
{
  "messages": [
    {"role": "user", "content": "Text to classify"},
    {"role": "assistant", "content": "language_code"}
  ]
}
```

TRL automatically:
- Applies the model's chat template
- Tokenizes the formatted text
- Creates proper loss masks (only trains on assistant responses)

### Training Configuration

- **Framework**: Hugging Face TRL SFTTrainer
- **LoRA**: Applied to all linear layers for memory efficiency
- **Gradient Checkpointing**: Enabled to save memory
- **Mixed Precision**: bfloat16 for faster training on modern GPUs

## Comparison to Tinker

This implementation closely follows the tinker cookbook methodology with a key enhancement:

**Same:**
- Teacher model: Qwen3-30B-A3B-Thinking (same as tinker)
- LoRA configuration: rank 32, alpha 16
- Learning rate: 1e-4
- Training epochs: 4
- Temperature: 0.15 (data generation)
- Prompt: Identical language classification prompt

**Enhanced:**
- **Student model**: Qwen3-30B-A3B-**Instruct** (non-thinking variant)
  - Removes thinking overhead for faster inference
  - Same model size, but direct responses without reasoning tokens
  - 20-30x faster than thinking model in production
- **Framework**: TRL (more accessible than tinker's internal framework)
- **Max length**: 4096 (student doesn't need long context)

**Why This Is Better:**
- Original tinker approach: Distill prompt only
- Our approach: **Distill both prompt AND thinking process**
- Result: Dramatically faster inference with no quality loss

## Expected Results

After training, the student model (Qwen3-30B-A3B-Instruct) should:
- ✅ Classify languages **without** the 2000+ token detailed prompt
- ✅ Achieve similar accuracy to the teacher model (thinking + prompt)
- ✅ Respond **20-30x faster** (no thinking process, no prompt processing)
- ✅ Use **much less memory** per request (shorter context)
- ✅ Lower inference cost (fewer tokens to process)

**Inference Speed Comparison:**

| Setup | Tokens Processed | Response Time | Cost |
|-------|------------------|---------------|------|
| **Teacher (Thinking + Prompt)** | ~2500 tokens | ~2-3 seconds | High |
| **Student (No Thinking, No Prompt)** | ~50 tokens | ~0.1 seconds | **50x cheaper** |

This dramatic speedup makes the distilled model practical for **production deployment** where latency and cost matter.

## Troubleshooting

### Out of Memory (OOM)

The 30B model requires a large H100 GPU (80GB). If you encounter OOM errors:

**Solutions:**
1. Reduce `per_device_train_batch_size` from 4 to 2 or 1
2. Reduce `max_length` from 2048 to 1024 or 512
3. Increase `gradient_accumulation_steps` to maintain effective batch size
4. Reduce `lora_rank` from 32 to 16 or 8

**Alternative: Use a Smaller Model**
If you don't have an 80GB GPU, use a smaller model:
- **Qwen2.5-7B-Instruct**: ~28GB memory, fits on most GPUs
- **Qwen2.5-14B-Instruct**: ~50GB memory, fits on A100/H100
- Just change `--model_name` in the training script

**Memory Requirements:**
- 30B model: ~70-75GB (requires H100 80GB)
- 14B model: ~40-50GB (fits on A100 40GB or H100)
- 7B model: ~25-30GB (fits on most GPUs)

### Data Generation Issues

If data generation fails or is slow:

1. Increase `tensor_parallel_size` to use more GPUs
2. Use the parallel script for H100x8: `bash create_data_h100x8.sh`
3. Reduce the dataset size for testing
4. Check GPU memory usage with `nvidia-smi`

### Training Not Converging

If the model doesn't learn:

1. Verify training data format is correct
2. Check that examples have valid language labels
3. Try increasing the number of training epochs
4. Adjust the learning rate (try 5e-5 or 2e-4)

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{askell2021general,
  title={A general language assistant as a laboratory for alignment},
  author={Askell, Amanda and others},
  journal={arXiv preprint arXiv:2112.00861},
  year={2021}
}

@article{snell2022learning,
  title={Learning by distilling context},
  author={Snell, Charlie and Klein, Dan and Zhong, Ruiqi},
  journal={arXiv preprint arXiv:2209.15189},
  year={2022}
}
```

And the Hugging Face TRL library:

```bibtex
@software{trl2024,
  title={TRL: Transformer Reinforcement Learning},
  author={TRL contributors},
  url={https://github.com/huggingface/trl},
  year={2024}
}
```

## License

This project follows the same license as the TRL library (Apache 2.0).

## Acknowledgments

- Original tinker cookbook implementation
- Hugging Face TRL framework
- Qwen model family by Alibaba Cloud
