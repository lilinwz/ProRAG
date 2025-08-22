import json
import glob
import os

def merge_json_lists(output_file):
    merged_list = []
    
    json_files = ['/home/v-zhaowan/zhaowang/rag/sample/data_mcts_500_filtered.json',
    '/home/v-zhaowan/zhaowang/rag/sample/data_mcts_1000_filtered.json',
    '/home/v-zhaowan/zhaowang/rag/sample/data_mcts_1500_filtered.json']
    print(f"找到以下文件进行合并: {json_files}")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_list.extend(data)
                else:
                    print(f"警告: '{file_path}' 的内容不是一个列表，已跳过。")
        except json.JSONDecodeError:
            print(f"警告: 无法解析 '{file_path}'，已跳过。")
        except Exception as e:
            print(f"处理 '{file_path}' 时发生错误: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文等字符正确显示
            # indent=4         使输出的JSON文件格式化，更易读
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        print(f"成功合并 {len(json_files)} 个文件到 '{output_file}'。")
        print(f"合并后的列表总共有 {len(merged_list)} 个项目。")
    except Exception as e:
        print(f"写入到 '{output_file}' 时发生错误: {e}")


output_filename = '/home/v-zhaowan/zhaowang/rag/rm/data.json'

merge_json_lists(output_filename)