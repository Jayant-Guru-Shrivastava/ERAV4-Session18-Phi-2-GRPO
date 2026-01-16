from datasets import load_dataset
import pandas as pd

def main():
    print("Loading dataset...")
    # Load the dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    
    # Convert to pandas for easier join operations (oasst1 is a tree structure)
    print("Processing dataframe...")
    df = ds.to_pandas()
    
    # Create a mapping from message_id to text
    id_to_text = df.set_index("message_id")["text"].to_dict()
    
    # Filter for assistant messages that have a parent
    assistant_msgs = df[
        (df["role"] == "assistant") & 
        (df["parent_id"].notna())
    ]
    
    formatted_data = []
    
    print(f"Found {len(assistant_msgs)} assistant messages. Constructing pairs...")
    
    for _, row in assistant_msgs.iterrows():
        parent_id = row["parent_id"]
        
        # Look up parent text (the prompt)
        if parent_id in id_to_text:
            prompt_text = id_to_text[parent_id]
            completion_text = row["text"]
            
            # Phi-2 Chat Format often uses similar to:
            # Instruct: <prompt>
            # Output: <completion>
            # But we will use the generic generic instruction format we defined
            formatted_item = {
                "prompt": f"### Instruction:\n{prompt_text}\n\n### Response:\n",
                "completion": completion_text
            }
            formatted_data.append(formatted_item)
            
    print(f"Generated {len(formatted_data)} training examples.")
    
    # Create new dataset from pairs
    from datasets import Dataset
    new_ds = Dataset.from_list(formatted_data)
    
    # Save processed dataset
    output_path = "./processed_oasst"
    new_ds.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    main()
