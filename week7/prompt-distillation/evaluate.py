"""
Evaluation script for the distilled prompt model.

This script evaluates the student model's performance on language classification
without providing the detailed prompt.

The student model (Qwen3-30B-A3B-Instruct) has been distilled from the teacher
(Qwen3-30B-A3B-Thinking) which used a 2000+ token prompt. After distillation,
the student responds directly without needing the prompt or thinking process.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def load_model(model_path: str, base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
    """Load the fine-tuned model with LoRA adapters."""
    print(f"Loading base model: {base_model}")
    print(f"This may take a few minutes for the 30B model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,  # Use 'dtype' instead of deprecated 'torch_dtype'
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer


def parse_language_label(response: str) -> Optional[str]:
    """Extract the language label from model response."""
    # Try to match common patterns
    patterns = [
        r"^([a-z]{2})$",  # Just "en", "fr", etc.
        r"^([a-z]{2})\s*$",  # With trailing whitespace
        r"Final Answer:\s*([a-z]{2})",  # With "Final Answer:" prefix
        r"Language:\s*([a-z]{2})",  # With "Language:" prefix
    ]
    
    response = response.strip().lower()
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # If response is short and looks like a language code
    if len(response) == 2 and response.isalpha():
        return response
    
    return None


def evaluate_model(
    model,
    tokenizer,
    test_sentences: List[str],
    ground_truth_labels: Optional[List[str]] = None,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> Dict:
    """Evaluate the distilled model on test sentences."""
    
    predictions = []
    correct = 0
    total = 0
    
    print("\nEvaluating model...")
    print("="*80)
    
    for idx, sentence in enumerate(test_sentences):
        # Format as user message (no system prompt!)
        messages = [
            {"role": "user", "content": sentence}
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse prediction
        pred_label = parse_language_label(response)
        predictions.append(pred_label)
        
        # Check correctness and show real-time progress
        if ground_truth_labels and idx < len(ground_truth_labels):
            gt_label = ground_truth_labels[idx]
            is_correct = pred_label == gt_label
            if is_correct:
                correct += 1
            total += 1
            
            # Show every sample in real-time
            status = "✓" if is_correct else "✗"
            status_color = "✓" if is_correct else "✗"
            
            # Truncate sentence for display
            display_sentence = sentence if len(sentence) <= 60 else sentence[:57] + "..."
            
            print(f"{status_color} [{idx+1:4d}/{len(test_sentences)}] "
                  f"Pred: {pred_label:>2s} | GT: {gt_label:>2s} | "
                  f"Acc: {correct}/{total} ({correct/total*100:5.1f}%) | "
                  f"{display_sentence}")
        else:
            # No ground truth - just show prediction
            display_sentence = sentence if len(sentence) <= 60 else sentence[:57] + "..."
            print(f"  [{idx+1:4d}/{len(test_sentences)}] "
                  f"Pred: {pred_label:>2s} | "
                  f"{display_sentence}")
    
    print("="*80)
    print(f"Evaluation completed: {len(test_sentences)} samples processed")
    
    # Calculate metrics
    results = {
        "predictions": predictions,
        "total": len(test_sentences),
        "predicted": sum(1 for p in predictions if p is not None),
        "unparseable": sum(1 for p in predictions if p is None),
    }
    
    if ground_truth_labels:
        results["accuracy"] = correct / total if total > 0 else 0.0
        results["correct"] = correct
        results["evaluated"] = total
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the distilled prompt model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/prompt_distillation_trl",
        help="Path to the trained LoRA adapters (default: ./models/prompt_distillation_trl)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Base student model name (non-thinking variant)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./example-data/multilingual.txt",
        help="Test sentences file (one per line)",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        default=None,
        help="Ground truth labels file (optional, one per line). If not provided, will try to load from training data.",
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="./data/prompt_distillation_lang.jsonl",
        help="Training data file to extract ground truth from (if ground_truth_file not provided)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_results.json",
        help="Where to save evaluation results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Load test data
    print(f"\nLoading test sentences from: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_sentences = [line.strip() for line in f if line.strip()]
    
    # Load ground truth
    ground_truth = None
    if args.ground_truth_file:
        print(f"Loading ground truth from: {args.ground_truth_file}")
        with open(args.ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth = [line.strip() for line in f if line.strip()]
    elif args.train_data_file and Path(args.train_data_file).exists():
        # Try to extract ground truth from training data
        print(f"Loading ground truth from training data: {args.train_data_file}")
        ground_truth = []
        sentence_to_label = {}
        
        with open(args.train_data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    if len(messages) >= 2:
                        user_content = messages[0].get("content", "")
                        assistant_content = messages[1].get("content", "")
                        sentence_to_label[user_content] = assistant_content
        
        # Match test sentences to ground truth labels
        for sentence in test_sentences:
            label = sentence_to_label.get(sentence)
            ground_truth.append(label if label else "unknown")
        
        matched = sum(1 for gt in ground_truth if gt != "unknown")
        print(f"Matched {matched}/{len(test_sentences)} test sentences to training data")
        
        if matched == 0:
            print("⚠️  Warning: No ground truth matched. Results will not show accuracy.")
            ground_truth = None
    
    # Limit samples if requested
    if args.max_samples:
        test_sentences = test_sentences[:args.max_samples]
        if ground_truth:
            ground_truth = ground_truth[:args.max_samples]
        print(f"Limiting evaluation to {args.max_samples} samples")
    
    print(f"Total test sentences: {len(test_sentences)}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_sentences=test_sentences,
        ground_truth_labels=ground_truth,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.base_model}")
    print(f"Adapter: {args.model_path}")
    print(f"\nPerformance:")
    print(f"  Total samples: {results['total']}")
    print(f"  Successfully predicted: {results['predicted']}")
    print(f"  Unparseable responses: {results['unparseable']}")
    print(f"  Parse rate: {results['predicted']/results['total']*100:.2f}%")
    
    if "accuracy" in results:
        print(f"\n  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Correct: {results['correct']}/{results['evaluated']}")
        print(f"\n💡 The model responds directly without the 2000+ token prompt!")
    
    # Save results
    output = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "test_file": args.test_file,
        "metrics": {
            k: v for k, v in results.items() if k != "predictions"
        },
    }
    
    # Optionally include predictions
    if args.output_file:
        output["predictions"] = results["predictions"]
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

