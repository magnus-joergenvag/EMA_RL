import json
import sys

def main(jsonl_path, txt_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, open(txt_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            obj = json.loads(line)
            # Assumes obj["messages"] is a list, and you want the "content" field from each item
            for message in obj.get("messages", []):
                content = message.get("content", "")
                f_out.write(content.replace("\n", "\\n") + "\n")  # save as one-liner per content

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jsonl_extract_content.py input.jsonl output.txt")
    else:
        main(sys.argv[1], sys.argv[2])