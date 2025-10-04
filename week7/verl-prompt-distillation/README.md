# Prompt Distillation with verl

This project reproduces the **prompt distillation** experiment from the [tinker cookbook](../tinker-cookbook/tinker_cookbook/recipes/prompt_distillation/) using the open-source [verl](https://github.com/volcengine/verl) framework.

## What is Prompt Distillation?

Prompt Distillation (also known as **context distillation**) is a training method that makes an LLM internalize a long and complex prompt into its parameters. After training, the model behaves as if it had been provided with the prompt, even without actually accessing it.

For example, we want to internalize this target prompt:

> "Classify the language of the provided text into these labels: en, fr, zh, ja ..."

After prompt distillation, the LLM will respond with only the language label without seeing the prompt:

```
Query: 一生、バンドしてくれる？
Response: ja
```

## Methodology

The method involves two stages:

1. **Data Generation (Teacher Model)**: A teacher model uses the detailed prompt to generate responses on a set of queries.
   - Teacher generates: `response = teacher(prompt, query)`
   
2. **Student Training (Distillation)**: A student model is fine-tuned to predict responses without accessing the prompt.
   - Student learns: `student(query) ≈ teacher(prompt, query)`

## Hyperparameters (Matching Tinker)

This implementation uses **exactly the same hyperparameters** as the tinker cookbook:

| Parameter | Value | Source |
|-----------|-------|--------|
| **Teacher Model** | Qwen3-30B-A3B-Thinking-2507 | For better accuracy |
| **Student Model** | Qwen3-4B-Instruct-2507 | Smaller for efficiency |
| **LoRA Rank** | 32 | tinker |
| **LoRA Alpha** | 16 | verl default |
| **Learning Rate** | 1e-4 | tinker |
| **LR Schedule** | linear | tinker |
| **Batch Size** | 128 | tinker |
| **Max Length** | 32768 | tinker |
| **Num Epochs** | 4 | tinker |
| **Temperature** | 0.15 | tinker (data generation) |
| **Weight Decay** | 0.01 | verl default |
| **Warmup Ratio** | 0.1 | verl default |
| **Gradient Clipping** | 1.0 | verl default |

*Note: We use [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) as teacher (matches tinker exactly) and [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) as student (smaller, more efficient for deployment).

## Dataset

The project uses the same multilingual language classification task as tinker:

- **Task**: Classify text into 13 language labels
- **Labels**: `ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot`
- **Source Data**: `../tinker-cookbook/example-data/multilingual.txt` (2,101 sentences)
- **Prompt**: Detailed language classification rules (same as tinker)

## Installation

### Prerequisites

1. Install verl framework:
```bash
cd ../verl
pip install -e .
```

2. Install additional dependencies:
```bash
cd ../verl-prompt-distillation
pip install -r requirements.txt
```

### System Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU acceleration)
- Recommended: 2+ GPUs with 24GB+ VRAM each

## Usage

### Step 1: Generate Training Data

Generate prompt distillation data using the teacher model:

```bash
# Option 1: Single instance (uses 4 GPUs with TP=4)
python create_data.py \
    --input_file ./example-data/multilingual.txt \
    --output_file ./data/prompt_distillation_lang.jsonl \
    --model_name Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --temperature 0.15 \
    --tensor_parallel_size 4

# Option 2: Parallel instances (uses ALL 8 GPUs - RECOMMENDED for H100x8)
bash create_data_h100x8_parallel.sh
```

**Options:**
- `--input_file`: Path to input sentences (one per line)
- `--output_file`: Where to save generated training data
- `--model_name`: Teacher model (Qwen3-30B-A3B-Thinking-2507 for better accuracy)
- `--temperature`: Sampling temperature (0.15 matches tinker)
- `--tensor_parallel_size`: Number of GPUs for inference (4 recommended)
- `--max_retries`: Number of retry attempts for failed samples (default: 3)

**For H100x8 users**: Use `create_data_h100x8_parallel.sh` to utilize all 8 GPUs (runs 2 instances in parallel, each with TP=4)

This will:
- Load sentences from the multilingual dataset
- Use the teacher model to generate language labels with the full prompt
- Save training data in JSONL format for verl

### Step 2: Train the Student Model

Fine-tune the student model on the distilled data:

```bash
bash train_sft.sh 2 ./models/prompt_distillation
```

**Arguments:**
- First arg: Number of GPUs to use (e.g., 2)
- Second arg: Where to save model checkpoints

The training script will:
- Load the generated distillation dataset (from 30B teacher)
- Fine-tune the 4B student model using LoRA (rank 32)
- Train for 4 epochs with batch size 128
- Use learning rate 1e-4 with linear schedule

### Step 3: Test Your Model

After training, you can test the distilled model:

```bash
python evaluate.py \
    --model_path ./models/prompt_distillation/checkpoint-final \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --test_file ../tinker-cookbook/example-data/multilingual.txt \
    --max_samples 100
```

## Project Structure

```
verl-prompt-distillation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── create_data.py                     # Data generation script (Step 1)
├── train_sft.sh                       # Training script (Step 2)
├── data/                              # Generated training data
│   └── prompt_distillation_lang.jsonl
└── models/                            # Trained model checkpoints
    └── prompt_distillation/
```

## Differences from Tinker

While we aim to match the tinker implementation as closely as possible, there are some differences:

1. **Framework**: Using verl (open source) instead of tinker (closed source)
2. **Teacher/Student Split**: We use a 30B teacher and 4B student (classic distillation setup)
   - Teacher: Qwen3-30B-A3B-Instruct-2507 (matches tinker's base model)
   - Student: Qwen3-4B-Instruct-2507 (smaller, more efficient)
   - Note: Tinker uses the same model as both teacher and student
3. **Inference Engine**: Using vLLM instead of tinker's internal engine
4. **Training Backend**: Using PyTorch FSDP2 via verl instead of tinker's backend

All **hyperparameters** remain identical to ensure reproducibility. Using different sized teacher/student models is actually a more common and practical prompt distillation scenario.

## Key Implementation Details

### Data Format

The generated data follows verl's multi-turn format:

```json
{
  "messages": [
    {"role": "user", "content": "Text in some language"},
    {"role": "assistant", "content": "en"}
  ]
}
```

### Training Configuration

- **Strategy**: FSDP2 (Fully Sharded Data Parallel)
- **LoRA**: Applied to all linear layers
- **Padding**: Uses remove padding for efficiency
- **Gradient Checkpointing**: Enabled to save memory

## Comparison to Original Paper

This implementation is based on:

1. **Askell et al. (2021)**: "A general language assistant as a laboratory for alignment"
2. **Snell et al. (2022)**: "Learning by distilling context"

The methodology follows the same two-stage approach described in these papers.

## Expected Results

After training, the student model should:
- Classify languages without the detailed prompt
- Achieve similar accuracy to the teacher model (with prompt)
- Respond much faster (no prompt processing overhead)
- Use less memory (shorter context)

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

**Data Generation (30B Teacher):**
1. Increase `tensor_parallel_size` to use more GPUs (e.g., 2 or 4)
2. Reduce batch size or use quantization (int8/int4)

**Training (4B Student):**
1. Reduce `micro_batch_size_per_gpu` in `train_sft.sh`
2. Reduce `max_length` (e.g., from 32768 to 16384)
3. Use more GPUs with `nproc_per_node`
4. Enable CPU offloading in FSDP config

### Slow Data Generation

If data generation is slow:

1. Increase `tensor_parallel_size` to use multiple GPUs
2. Use a smaller teacher model
3. Reduce the dataset size for testing

### Training Not Converging

If the model doesn't learn:

1. Check that data generation completed successfully
2. Verify the data format is correct
3. Try increasing the number of training epochs
4. Adjust the learning rate

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

And the verl framework:

```bibtex
@software{verl2024,
  title={verl: Volcano Engine Reinforcement Learning for LLM},
  author={verl contributors},
  url={https://github.com/volcengine/verl},
  year={2024}
}
```

## License

This project follows the same license as the verl framework.

## Acknowledgments

- Original tinker cookbook implementation
- verl framework by Volcano Engine
- Qwen model family by Alibaba Cloud

