import json
import re
import openai
import argparse
from tqdm import tqdm
import os
from multiprocessing import Pool
import math
import logging

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_boxed(text)

def extract_boxed(text):
    pattern = r"\\boxed\{([A-J])\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def setup_logger(process_id):
    os.makedirs(f"logs/{args.model}", exist_ok=True)
    logger = logging.getLogger(f"process_{process_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"logs/{args.model}/process_{process_id}.log", mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def process_batch(batch_data, args, process_id):
    """处理一批数据的函数"""
    logger = setup_logger(process_id)

    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        initial_prompt_content = fi.read()

    batch_results = []
    batch_correct = 0

    for idx, item in enumerate(batch_data):
        try:
            prompt = "<｜begin▁of▁sentence｜><｜User｜>"
            prompt += initial_prompt_content
            subject = item["category"]
            gt_answer = item["ground_truth_answer"]
            prompt = prompt.replace("{$}", subject) + "\n"
            prompt += f"Question:\n"
            question = item["question"]
            options = item["options"]
            prompt += question + "\n"
            prompt += "Options:\n"
            for i, opt in enumerate(options):
                prompt += "{}. {}\n".format(choices[i], opt)
            slow_thinking_answer = item["predicted_answer1"]
            fast_thinking_answer = item["predicted_answer2"]
            prompt += f"""<｜Assistant｜>\n<think>\nI think there are two candidate answers\n\n{slow_thinking_answer}\n\nand\n\n{fast_thinking_answer}\n\nfor this question. One of them is correct or both are wrong. I need to first verfy them. If both are wrong, I need to rethink step by step and avoid making the same mistake to select the correct letter choice."""

            response = client.completions.create(
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=2400
            )
            output = response.choices[0].text.strip()
            pred_answer = extract_answer(output)
            is_correct = (pred_answer == gt_answer)
            if is_correct:
                batch_correct += 1

            result = {
                "question_id": item["question_id"],
                "problem": question,
                "model_output": output,
                "options": options,
                "predicted_answer": pred_answer,
                "ground_truth_answer": gt_answer,
                "correct": is_correct,
                "slow_thinking_answer": slow_thinking_answer,
                "fast_thinking_answer": fast_thinking_answer
            }
            batch_results.append(result)

        except Exception as e:
            logger.error(f"Error in process {process_id}, question_id {item.get('question_id', 'N/A')}: {str(e)}", exc_info=True)
            continue

    logger.info(f"Process {process_id} finished: {batch_correct}/{len(batch_data)} correct.")
    return batch_correct, len(batch_data), batch_results

def split_data(data, k):
    """将数据分割成k份"""
    n = len(data)
    batch_size = math.ceil(n / k)
    return [data[i * batch_size: min((i + 1) * batch_size, n)] for i in range(k) if i * batch_size < n]

def evaluate_model(args):
    if args.prompt_type == "JointThinking-thinking-middle-open":
        if not args.reference_ideas:
            raise ValueError("For JointThinking prompt type, --reference_ideas must be provided.")

        with open(args.reference_ideas, "r") as f:
            data = [json.loads(line) for line in f]

        data_batches = split_data(data, args.k)
        print(f"Split data into {len(data_batches)} batches for parallel processing")

        with Pool(processes=args.k) as pool:
            task_args = [(batch, args, i) for i, batch in enumerate(data_batches)]
            results_list = []

            with tqdm(total=len(data), desc="Processing samples", unit="sample") as pbar:
                async_results = [pool.apply_async(process_batch, task_arg) for task_arg in task_args]

                for i, async_result in enumerate(async_results):
                    batch_correct, batch_total, batch_results = async_result.get()
                    results_list.append((batch_correct, batch_total, batch_results))
                    pbar.update(batch_total)
                    pbar.set_postfix({
                        'Batch': f"{i+1}/{len(async_results)}",
                        'Processed': f"{sum(r[1] for r in results_list)}/{len(data)}"
                    })

        total_correct = sum(result[0] for result in results_list)
        total_samples = sum(result[1] for result in results_list)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        results = []
        for result in results_list:
            results.extend(result[2])

        print(f"\n✅ Accuracy on MMLU-pro JointThinking Part: {accuracy * 100:.2f}% ({total_correct}/{total_samples})")

        if hasattr(args, "save_path") and args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            with open(args.save_path, "w", encoding="utf-8") as fout:
                for res in results:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.write(json.dumps({
                    "summary": {
                        "accuracy": accuracy,
                        "correct": total_correct,
                        "total": total_samples
                    }
                }, ensure_ascii=False) + "\n")
            print(f"\n💾 Results saved to: {args.save_path}")
    else:
        raise ValueError(f"Unsupported prompt_type: {args.prompt_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on GPQA main using vLLM OpenAI-compatible API.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="OpenAI API key")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="vLLM OpenAI API base URL")
    parser.add_argument("--data_path", type=str, required=True, help="Path to GPQA dataset (.jsonl)")
    parser.add_argument("--prompt_type", type=str, choices=["direct", "nothinking", "JointThinking-thinking-middle-open"], default="direct", help="Prompt style")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for output")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save evaluation results as jsonl")
    parser.add_argument('--reference_ideas', type=str, default=None, help='Path to the reference ideas file (jsonl) for JointThinking prompt type')
    parser.add_argument('--k', type=int, default=4, help='Number of processes for parallel processing')
    args = parser.parse_args()

    evaluate_model(args)
