import csv
import io
import os
import random
import wave
import sys

def main():
    csv_path = "/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme1.csv"
    audio_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios/"
    output_wav = "/tmp/demo_with_ground_truth.wav"
    target_duration = 120.0
    
    # 1. Read CSV and group by Interview_Class to ensure a mix
    clips_by_class = {"Excited": [], "Unconfident": [], "Neutral_3Class": []}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.join(audio_dir, row["FileName"])
            if os.path.exists(path):
                label = row["Interview_Class"]
                if label in clips_by_class:
                    clips_by_class[label].append({
                        "path": path,
                        "label": label,
                        "text": row["Text"],
                        "filename": row["FileName"]
                    })
    
    # 2. Select clips randomly but alternating classes to make it dynamic
    random.seed(123) # Fixed seed for reproducible output
    
    selected_clips = []
    current_duration = 0.0
    
    # Interleave classes
    classes = ["Neutral_3Class", "Excited", "Unconfident", "Neutral_3Class", "Excited", "Unconfident"]
    class_idx = 0
    
    all_frames = b""
    params = None
    
    ground_truth = []
    
    while current_duration < target_duration:
        cls = classes[class_idx % len(classes)]
        class_idx += 1
        
        if not clips_by_class[cls]:
            continue
            
        clip = random.choice(clips_by_class[cls])
        
        try:
            with wave.open(clip["path"], "rb") as wf:
                if params is None:
                    params = wf.getparams()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                dur = n_frames / wf.getframerate()
                
                # Record ground truth
                ground_truth.append({
                    "start": current_duration,
                    "end": current_duration + dur,
                    "label": clip["label"],
                    "text": clip["text"]
                })
                
                all_frames += frames
                current_duration += dur
                selected_clips.append(clip)
        except Exception as e:
            continue
            
    # 3. Write output file
    with wave.open(output_wav, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(all_frames)
        
    print(f"✅ Generated {current_duration:.1f}s demo audio: {output_wav}\n")
    print("================================================================================")
    print("  📋 GROUND TRUTH CHEAT SHEET (What we actually put in the audio)")
    print("================================================================================")
    for i, gt in enumerate(ground_truth):
        print(f"  [{i+1:2d}] {gt['start']:5.1f}s ~ {gt['end']:5.1f}s | {gt['label']:14s}")
        print(f"        📝 \"{gt['text']}\"\n")

if __name__ == "__main__":
    main()
