import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualize_debug(img_path, refs, output_path):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at {img_path}")
        return
    
    img = Image.open(img_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    
    # Color list for different refs
    colors = ['lime', 'cyan', 'yellow', 'magenta', 'orange']

    for i, ref in enumerate(refs):
        label = ref["sentence"]
        if "segmentation" not in ref or not ref["segmentation"]:
            continue
        seg = ref["segmentation"][0]
        color = colors[i % len(colors)]
        
        # Process points
        points = np.array(seg).reshape(-1, 2)
        
        # Calculate Bbox
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        width, height = xmax - xmin, ymax - ymin

        # 1. Draw Segmentation (Polygon)
        polygon = patches.Polygon(points, closed=True, linewidth=2, 
                                  edgecolor=color, facecolor='none', alpha=0.8, 
                                  label=f"Seg: {label}")
        ax.add_patch(polygon)
        
        # 2. Draw Bbox (dashed line)
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, 
                                 edgecolor='red', facecolor='none', linestyle='--', 
                                 label=f"Bbox: {label}")
        ax.add_patch(rect)
        
        # 3. Add Label text near the bbox
        ax.text(xmin, ymin - 5, f"{i+1}: {label}", color='red', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Details
    ax.set_title(f"Debug: {os.path.basename(img_path)}\nDetected {len(refs)} artifacts")
    ax.legend(loc='upper right', fontsize='x-small')
    plt.axis('off')

    # Save result
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # Close the figure to free memory
    print(f"Debug image saved to {output_path}")


def process_synthscars(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        all_items = json.load(f)

    transformed_dataset = []

    for global_idx, entry in enumerate(all_items):
        # 1. 提取原始数据
        original_key = list(entry.keys())[0]
        data = entry[original_key]
        
        img_name = data["img_file_name"]
        caption = data["caption"]
        refs = data["refs"]

        # 2. 构造外部 bbox 字典和 answer 行
        bbox_metadata = {}
        answer_lines = []

        for i, ref in enumerate(refs):
            sentence = ref["sentence"]
            explanation = ref["explanation"]
            seg = ref["segmentation"][0]
            points = np.array(seg).reshape(-1, 2)
            
            # 计算原始像素坐标 [ymin, xmin, ymax, xmax]
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            current_bbox = [round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)]
            
            # 存储元数据：{ 索引: { 标签, 坐标, 解释 } }
            bbox_metadata[str(i)] = {
                "label": sentence,
                "bbox": current_bbox,
                "explanation": explanation
            }
            
            # --- 使用数字索引作为占位符 ---
            bbox_tag = f"bbox_{i}"
            line = (
                f"{i+1}. <|object_ref_start|>{sentence}<|object_ref_end|> "
                f"at <|box_start|>{{{bbox_tag}}}<|box_end|>: {explanation}"
            )
            answer_lines.append(line)

        # 3. 构造对话部分 (纯 Assistant)
        intro_text = "Detected artifacts as follows:"
        full_answer = f"<answer>\n{intro_text}\n" + "\n".join(answer_lines) + "\n</answer>"
        
        conversations = [
            {
                "from": "assistant",
                "value": f"<think>\n{caption}\n</think>\n{full_answer}"
            }
        ]

        # 4. 组装最终结构
        new_item = {
            "id": f"item_{global_idx}",
            "image": img_name,
            "bbox": bbox_metadata, 
            "refs": refs,
            "conversations": conversations
        }
        transformed_dataset.append(new_item)

        print(f"Processed item {global_idx + 1}/{len(all_items)}", end='\r')

        # # --- Debug Visualization ---
        # img_dir = os.path.dirname(os.path.dirname(input_json_path)) + "/images"
        # img_path = os.path.join(img_dir, img_name)
        # debug_output_path = f"debug_{img_name}"
        # visualize_debug(img_path, refs, debug_output_path)
        
        # print(f"Bbox and seg visualization: {debug_output_path}")
        # import ipdb; ipdb.set_trace()

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_dataset, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 转换完成！共处理了 {len(transformed_dataset)} 条目。")

# 执行转换
# transform_to_indexed_template("train.json", "final_indexed_dataset.json")


if __name__ == "__main__":
    process_synthscars("datasets/SynthScars/train/annotations/train.json", "datasets/SynthScars/train/annotations/train_bbox.json")



