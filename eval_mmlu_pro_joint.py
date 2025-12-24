import json
import re
import openai
import argparse
from tqdm import tqdm
import os

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
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

def evaluate_model(args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    
    client = openai.OpenAI(
                api_key=args.api_key,
                base_url=args.api_base,
            )
    
    if args.prompt_type in ["JointThinking-thinking-middle-open"]:
        if not args.reference_ideas:
            raise ValueError("For JointThinking prompt type, --reference_ideas must be provided.")
        with open(args.reference_ideas, "r") as f:
            data = [json.loads(line) for line in f]
            
        correct = 0
        total = 0
        results = []
        for item in tqdm(data, desc="JointThinking"):
            prompt = "<｜begin▁of▁sentence｜><｜User｜>"
            with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
                for line in fi.readlines():
                    prompt += line
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
            try:
                response = client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=14000,
                    timeout=2400
                )
                output = response.choices[0].text.strip()
                pred_answer = extract_answer(output)
                
                is_correct = (pred_answer == gt_answer)
                if is_correct:
                    correct += 1

                results.append({
                    "question_id": item["question_id"],
                    "problem": question,
                    "model_output": output,
                    "options": options,
                    "predicted_answer": pred_answer,
                    "ground_truth_answer": gt_answer,
                    "correct": is_correct
                })

            except Exception as e:
                print(f"Error: {e}")
                continue

            total += 1
            if total <= 10:
                print(f"Problem: {question}")
                print(f"Model Output: {output}")
                print(f"Slow Thinking Answer: {slow_thinking_answer}")
                print(f"Fast Thinking Answer: {fast_thinking_answer}")
                print(f"Predicted Answer: {pred_answer}")
                print(f"Ground Truth Answer: {gt_answer}")
                print(f"Correct: {is_correct}") 
                print("-" * 50)

        accuracy = correct / total if total > 0 else 0
        print(f"\n✅ Accuracy on MMLU-pro JointThinking Part: {accuracy * 100:.2f}% ({correct}/{total})")

        # 保存结果到文件
        if hasattr(args, "save_path") and args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            
            with open(args.save_path, "w", encoding="utf-8") as fout:
                for res in results:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.write(json.dumps({
                    "summary": {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total
                    }
                }, ensure_ascii=False) + "\n")
            
        return
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

    args = parser.parse_args()
    evaluate_model(args)
