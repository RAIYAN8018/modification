import json
import time
from chatbot_setup import process_records_parallel, gpt_translate_text

def load_json_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data

def save_json_file(data, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    input_file = input("Enter path to input JSON file: ").strip()
    data = load_json_file(input_file)
    
    print(f"Loaded {len(data)} records.")
    
    start_time = time.time()
    
    # Process records
    print("Processing records...")
    processed = process_records_parallel(data, max_workers=5)
    
    # Translate to English
    print("Translating to English...")
    processed_en = [gpt_translate_text(rec, "English") for rec in processed]
    
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f} seconds.")
    
    # Summary
    valid = sum(1 for r in processed if r.get("_validation", {}).get("valid") is True)
    corrected = sum(1 for r in processed if r.get("_validation", {}).get("corrected") is True)
    failed = sum(1 for r in processed if not r.get("_validation", {}).get("valid", False) and not r.get("_validation", {}).get("corrected", False))
    
    print("Summary:")
    print(f"Total records: {len(processed)}")
    print(f"Valid (no changes): {valid}")
    print(f"Corrected by AI: {corrected}")
    print(f"Failed to correct/flagged: {failed}")
    
    # Save output files
    save_json_file(processed, "corrected.json")
    save_json_file(processed_en, "corrected_en.json")
    
    print("Output files saved as 'corrected.json' and 'corrected_en.json'.")

if __name__ == "__main__":
    main()
