import os
import csv
import time
import requests
import random

API_URL = "http://0.0.0.0:8001/v1/emotion/classify-batch"
CSV_PATH = "/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme1.csv"
AUDIO_DIR = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios/"

def test():
    # 1. Read CSV and find 30 valid files
    print("Loading CSV and looking for files...")
    valid_samples = []
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return
        
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["FileName"]
            path = os.path.join(AUDIO_DIR, fname)
            if os.path.exists(path):
                # Find ground truth label
                label = None
                if float(row["Excited"]) == 1.0:
                    label = "Excited"
                elif float(row["Unconfident"]) == 1.0:
                    label = "Unconfident"
                elif float(row["Neutral_3Class"]) == 1.0:
                    label = "Neutral_3Class"
                
                if label:
                    valid_samples.append({
                        "path": path,
                        "filename": fname,
                        "text": row["Text"],
                        "label": label
                    })
                    
    print(f"Total valid files found in CSV: {len(valid_samples)}")
    
    if len(valid_samples) == 0:
        print("No valid files found with audio matching CSV.")
        return
        
    # Randomly sample 30 files if we have more than 30
    if len(valid_samples) > 30:
        print("Randomly sampling 30 files for testing...")
        valid_samples = random.sample(valid_samples, 30)
    else:
        print(f"Testing with all {len(valid_samples)} available files.")
        
    # 2. Split into batches of 15 (since MAX_BATCH=16)
    batch_size = 15
    batches = [valid_samples[i:i + batch_size] for i in range(0, len(valid_samples), batch_size)]
    
    correct = 0
    total = 0
    total_time = 0
    
    for b_idx, batch in enumerate(batches):
        print(f"\nProcessing Batch {b_idx+1}/{len(batches)} (Size: {len(batch)})...")
        
        # Prepare files and forms
        files = []
        # Form data list of tuples for texts
        # requests requires a list of tuples for duplicate keys in Form
        data_fields = []
        file_handles = []
        
        for item in batch:
            fh = open(item["path"], "rb")
            file_handles.append(fh)
            files.append(("files", (item["filename"], fh, "audio/wav")))
            # data_fields.append(("texts", item["text"]))
            
        t0 = time.perf_counter()
        try:
            resp = requests.post(API_URL, files=files, data=data_fields)
            latency = (time.perf_counter() - t0) * 1000
            total_time += latency
            
            if resp.status_code != 200:
                print(f"Error: {resp.status_code} - {resp.text}")
                continue
                
            data = resp.json()
            results = data["results"]
            
            for i, item in enumerate(batch):
                pred = results[i]["primary_label"]
                gt = item["label"]
                total += 1
                if pred == gt:
                    correct += 1
                print(f"  File: {item['filename']} | GT: {gt:<15} | Pred: {pred:<15} | {'✅' if pred==gt else '❌'}")
                
        except Exception as e:
            print(f"Request failed: {e}")
        finally:
            for fh in file_handles:
                fh.close()
                
    print("\n" + "="*50)
    print(f"Total Samples: {total}")
    if total > 0:
        print(f"Accuracy     : {correct/total*100:.2f}% ({correct}/{total})")
    print(f"Total Time   : {total_time:.1f} ms")
    if total > 0:
        print(f"Avg per item : {total_time/total:.1f} ms")
    print("="*50)

if __name__ == "__main__":
    test()
