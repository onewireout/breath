#!/usr/bin/env python3
import os
import argparse
from pydub import AudioSegment

def parse_audacity_labels(label_file):
    """
    Parses an Audacity labels text file.
    Format: start_time \t end_time \t label_name
    """
    labels = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                start = float(parts[0])
                end = float(parts[1])
                label_name = parts[2] if len(parts) > 2 else "Breath"
                labels.append({
                    'start': start,
                    'end': end,
                    'name': label_name
                })
    return labels

def process_audio(wav_file, label_file, output_dir):
    print("\n--- Starting Audio Splitting ---")
    audio = AudioSegment.from_wav(wav_file)
    audio_length_seconds = len(audio) / 1000.0
    
    dataset_num = 1
    training_images_dir = os.path.join(output_dir, "Training-Images")
    if os.path.exists(training_images_dir):
        for root, _, files in os.walk(training_images_dir):
            for file in files:
                if file.startswith("Dataset_"):
                    parts = file.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        dataset_num = max(dataset_num, int(parts[1]) + 1)
                        
    base_wav_name = f"Dataset_{dataset_num}"
    
    labels = parse_audacity_labels(label_file)
    
    input_dir = os.path.join(output_dir, "Input")
    breath_dir = os.path.join(input_dir, "Breath")
    speaking_dir = os.path.join(input_dir, "Speaking")
    
    os.makedirs(breath_dir, exist_ok=True)
    os.makedirs(speaking_dir, exist_ok=True)
    
    # Labels may be out of order
    labels.sort(key=lambda x: x['start'])
    
    speaking_segments = []
    last_end = 0.0
    
    breath_count = 0
    for i, label in enumerate(labels):
        start = label['start']
        end = label['end']
        
        if start >= end:
            continue
            
        center_time = (start + end) / 2.0
        
        # Augmentation: 5 shifted 0.5s windows per breath
        shifts = [-0.15, -0.05, 0.0, 0.05, 0.15]
        
        for shift in shifts:
            window_start = center_time + shift - 0.25
            window_end = center_time + shift + 0.25
            
            # Keep window within file bounds
            if window_start < 0:
                window_start = 0.0
                window_end = 0.5
            if window_end > audio_length_seconds:
                window_end = audio_length_seconds
                window_start = max(0.0, audio_length_seconds - 0.5)
                
            breath_audio = audio[window_start * 1000: window_end * 1000]
            
            # Pad with silence if the file is under 0.5s
            if len(breath_audio) < 500:
                silence = AudioSegment.silent(duration=500 - len(breath_audio))
                breath_audio = breath_audio + silence
                
            breath_count += 1
            breath_filename = os.path.join(breath_dir, f"{base_wav_name}_Breath_{breath_count:04d}.wav")
            breath_audio.export(breath_filename, format="wav")
            
        print(f"\rExporting Breath clips... {i+1}/{len(labels)} (Generated {breath_count})", end="", flush=True)
        
        # Gap before this breath is Speaking
        if start > last_end:
            speaking_segments.append({'start': last_end, 'end': start})
            
        last_end = max(last_end, end)
        
    print()
        
    # Speaking after the last breath
    if last_end < audio_length_seconds:
        speaking_segments.append({'start': last_end, 'end': audio_length_seconds})
        
    speaking_count = 0
    for i, seg in enumerate(speaking_segments):
        start = seg['start']
        end = seg['end']
        
        # Cut into 0.5s blocks
        current_start = start
        while current_start + 0.5 <= end:
            speaking_audio = audio[current_start * 1000: (current_start + 0.5) * 1000]
            speaking_count += 1
            speaking_filename = os.path.join(speaking_dir, f"{base_wav_name}_Speaking_{speaking_count:04d}.wav")
            speaking_audio.export(speaking_filename, format="wav")
            print(f"\rExporting Speaking clips... {speaking_count}", end="", flush=True)
            current_start += 0.5

    print(f"\nCompleted {breath_count} Breaths and {speaking_count} Speaking clips!")

if __name__ == "__main__":
    import sys
    
    # Arguments can come in either order (wav and label file)
    wav_file = None
    label_file = None
    
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".wav"):
            wav_file = arg
        elif arg.lower().endswith(".txt"):
            label_file = arg
            
    if not wav_file or not label_file:
        print("Usage: Drag and drop BOTH the .wav file and the .txt label file onto this script.")
        print("Alternatively, run from terminal:")
        print("python split_audio.py <audio.wav> <labels.txt>")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        process_audio(wav_file, label_file, out_dir)
        
        print("\n--- Starting Mel-Spectrogram Conversion ---")
        import convert_to_mel
        input_directory = os.path.join(out_dir, "Input")
        output_directory = os.path.join(out_dir, "Training-Images")
        convert_to_mel.create_mel_spectrograms(input_directory, output_directory)
        
        print("\n--- Starting CNN Training ---")
        import train_cnn
        save_model_path = os.path.join(out_dir, 'best_mel_cnn.pth')
        train_cnn.main(data_dir=output_directory, save_path=save_model_path)
        
        import shutil
        if os.path.exists(input_directory):
            shutil.rmtree(input_directory)
            
        print("\nSuccessfully finished all processing!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    input("\nPress Enter to exit...")

