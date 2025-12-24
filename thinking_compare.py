import os
import json

def compare_json_folders(folder1, folder2, output_file="inconsistent_results.jsonl"):
    # 创建全局列表用于存储不一致的结果
    global_inconsistent_list = []
    global_count = 0
    global_correct_count = 0

    # 统计变量
    both_true = 0
    both_false = 0
    file1_true_file2_false = 0
    file1_false_file2_true = 0
    both_true_equivalent = 0  # 思考和nothinking都正确且答案相同
    both_false_equivalent = 0  # 思考和nothinking都错误且答案相同

    # 获取两个文件夹中的所有json文件
    files1 = [f for f in os.listdir(folder1) if f.endswith('.json')]
    files2 = [f for f in os.listdir(folder2) if f.endswith('.json')]

    # 检查文件数量是否相同
    if len(files1) != len(files2):
        print(f"错误：文件夹 {folder1} 和 {folder2} 中的json文件数量不一致")
        return global_inconsistent_list

    # 检查文件名是否一一对应
    files1_sorted = sorted(files1)
    files2_sorted = sorted(files2)

    if files1_sorted != files2_sorted:
        print("错误：文件夹中的json文件名称不一致")
        print("文件夹1中的文件:", files1_sorted)
        print("文件夹2中的文件:", files2_sorted)
        return global_inconsistent_list

    # 遍历每一对文件进行比较
    for filename in files1_sorted:
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)

        # 读取第一个文件的内容
        with open(file1_path, 'r', encoding='utf-8') as f1:
            try:
                thinking_list = json.load(f1)
            except json.JSONDecodeError:
                print(f"错误：无法解析文件 {file1_path} 的JSON内容")
                continue

        # 读取第二个文件的内容
        with open(file2_path, 'r', encoding='utf-8') as f2:
            try:
                nothinking_list = json.load(f2)
            except json.JSONDecodeError:
                print(f"错误：无法解析文件 {file2_path} 的JSON内容")
                continue

        # 检查两个列表的长度是否一致
        if len(thinking_list) != len(nothinking_list):
            print(f"错误：文件 {filename} 中的两个列表长度不一致")
            print(f"文件1中的列表长度: {len(thinking_list)}")
            print(f"文件2中的列表长度: {len(nothinking_list)}")
            continue
        
        global_count += len(thinking_list)
        # 比较每个索引处的答案
        for index in range(len(thinking_list)):
            item1 = thinking_list[index]
            item2 = nothinking_list[index]

            thinking_pred_answer = item1["pred"]
            nothinking_pred_answer = item2["pred"]
            ground_truth_answer = item1["answer"]  # 正确答案的索引
            category = item1["category"]
            question_id = item1["question_id"]
            question = item1["question"]
            options = item1["options"]
            predicted_full_answer1 = options[item1["answer_index"]]
            predicted_full_answer2 = options[item2["answer_index"]]
            
            # 判断答案是否正确(比较预测的答案索引与正确答案的索引)
            thinking_is_correct = (thinking_pred_answer == ground_truth_answer)
            nothinking_is_correct = (nothinking_pred_answer == ground_truth_answer)
            
            # 统计
            if thinking_is_correct and nothinking_is_correct:
                both_true += 1
                both_true_equivalent += 1  # 因为都正确，所以答案必然相同
            elif not thinking_is_correct and not nothinking_is_correct:
                both_false += 1
                # 检查答案是否等价(比较两个预测的完整答案文本)
                if thinking_pred_answer == nothinking_pred_answer:
                    both_false_equivalent += 1
            elif thinking_is_correct and not nothinking_is_correct:
                file1_true_file2_false += 1
            elif not thinking_is_correct and nothinking_is_correct:
                file1_false_file2_true += 1
            
            # 比较答案是否一致(你的原始逻辑)
            if thinking_pred_answer != nothinking_pred_answer:
                global_correct_count += 1
                inconsistent_item = {
                    "question_id": question_id,
                    "question": question,
                    "category": category,
                    "options": options,
                    "predicted_answer1": predicted_full_answer1,
                    "predicted_answer2": predicted_full_answer2,
                    "ground_truth_answer": ground_truth_answer  # 使用完整答案文本
                }
                global_inconsistent_list.append(inconsistent_item)

    # 将结果保存为JSONL文件
    if global_inconsistent_list:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in global_inconsistent_list:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"发现 {len(global_inconsistent_list)} 处不一致，结果已保存到 {output_file}")
    else:
        print("所有文件中的答案都一致")

    # 打印统计信息
    print("\n统计结果:")
    print(f"Both true (两个模型都正确): {both_true}")
    print(f"Both false (两个模型都错误): {both_false}")
    print(f"File1 true, File2 false (模型1正确但模型2错误): {file1_true_file2_false}")
    print(f"File1 false, File2 true (模型1错误但模型2正确): {file1_false_file2_true}")
    print(f"Both true and equivalent (两个模型都正确且答案相同): {both_true_equivalent}")
    print(f"Both false and equivalent (两个模型都错误且答案相同): {both_false_equivalent}")
    
    # 计算并打印一些额外信息
    total_comparisons = both_true + both_false + file1_true_file2_false + file1_false_file2_true
    if total_comparisons > 0:
        print(f"\n总计比较次数: {total_comparisons}")
        print(f"模型1正确率: {both_true + file1_true_file2_false} / {total_comparisons} = {(both_true + file1_true_file2_false)/total_comparisons:.2%}")
        print(f"模型2正确率: {both_true + file1_false_file2_true} / {total_comparisons} = {(both_true + file1_false_file2_true)/total_comparisons:.2%}")

    return global_inconsistent_list

if __name__ == "__main__":
    # 获取用户输入的两个文件夹路径
    folder1_path = "./eval_results/DeepSeek-R1-Distill-Qwen-1.5B/CoT/all"
    folder2_path = "./eval_results/DeepSeek-R1-Distill-Qwen-1.5B-nothinking/CoT/all"

    # 可选：让用户指定输出文件路径
    output_file = "./thinking_compare/mmlu-diff-r1-1_5b.jsonl"

    # 检查路径是否存在
    if not os.path.isdir(folder1_path):
        print(f"错误：文件夹 {folder1_path} 不存在")
    elif not os.path.isdir(folder2_path):
        print(f"错误：文件夹 {folder2_path} 不存在")
    else:
        # 执行比较
        result = compare_json_folders(folder1_path, folder2_path, output_file)

        # 在控制台输出简要结果
        if not result:
            print("所有文件中的答案都一致")
        else:
            print(f"发现 {len(result)} 处不一致，详细结果已保存到 {output_file}")