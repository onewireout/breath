#!/usr/bin/env python3
import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydub import AudioSegment
from torchvision import transforms

from train_cnn import SpectrogramCNN

def run_inference(wav_file, model_path, output_dir, threshold=0.75):
    print(f"\n--- Starting Inference on {os.path.basename(wav_file)} ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}.")
        print("Make sure you have trained the model first!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Class 0 = Breath, class 1 = Speaking
    model = SpectrogramCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Must match the training preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cmap = plt.get_cmap('magma')
    
    print("Loading audio...")
    waveform, sr = torchaudio.load(wav_file)
    waveform = waveform.to(device)
    
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128, f_max=8000.0).to(device)
    amp_transform = T.AmplitudeToDB().to(device)
    
    # Sliding window: 0.5s wide, 0.1s hop
    window_sec = 0.5
    hop_sec = 0.1
    
    audio_full = AudioSegment.from_wav(wav_file)
    total_duration = len(audio_full) / 1000.0
    
    predictions = []
    current_time = 0.0
    
    print(f"Processing ~{int(total_duration / hop_sec)} sliding windows...")
    
    batch_size = 128
    batch_chunks = []
    batch_times = []
    
    while current_time + window_sec <= total_duration:
        expected_len = int(round(window_sec * sr))
        start_sample = int(round(current_time * sr))
        end_sample = start_sample + expected_len
        
        chunk = waveform[:, start_sample:end_sample]
        
        # Pad the last chunk if it's short
        if chunk.shape[1] < expected_len:
            pad = torch.zeros((chunk.shape[0], expected_len - chunk.shape[1])).to(device)
            chunk = torch.cat((chunk, pad), dim=1)
        elif chunk.shape[1] > expected_len:  # off-by-one guard
            chunk = chunk[:, :expected_len]
            
        batch_chunks.append(chunk)
        batch_times.append(current_time)
        current_time += hop_sec
        sys.stdout.write(f"\rQueueing chunks: {current_time:.1f}s / {total_duration:.1f}s")
        sys.stdout.flush()
        
    print("\nQueueing complete. Starting batched inference...")
    
    num_batches = len(batch_chunks) // batch_size + (1 if len(batch_chunks) % batch_size != 0 else 0)
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(batch_chunks))
        
        chunks = batch_chunks[start_idx:end_idx]
        times = batch_times[start_idx:end_idx]
        
        stacked_chunks = torch.stack(chunks)
        
        mel_specs = mel_transform(stacked_chunks)
        mel_specs_db = amp_transform(mel_specs)
        specs = mel_specs_db.cpu().numpy()
        
        img_tensors = []
        for i in range(len(chunks)):
            spec = specs[i][0]
            spec_min = spec.min()
            spec_max = spec.max()
            if spec_max - spec_min > 0:
                spec_normalized = (spec - spec_min) / (spec_max - spec_min)
            else:
                spec_normalized = np.zeros_like(spec)
                
            spec_normalized = np.flipud(spec_normalized)
            spec_rgba = cmap(spec_normalized) 
            spec_rgb = (spec_rgba[:, :, :3] * 255).astype(np.uint8)
            
            image = Image.fromarray(spec_rgb, mode='RGB')
            img_tensor = img_transform(image)
            img_tensors.append(img_tensor)
            
        batched_imgs = torch.stack(img_tensors).to(device)
        
        with torch.no_grad(), torch.autocast(device_type=device.type):
            output = model(batched_imgs)
            probs = torch.softmax(output, dim=1)
            
        for i in range(len(chunks)):
            prob_breath = probs[i][0].item()
            prob_speaking = probs[i][1].item()
            
            if prob_breath >= 0.80:
                label = "Breath"
            else:
                label = "Speaking"
                
            predictions.append({
                'start': times[i],
                'end': times[i] + window_sec,
                'label': label
            })
            
        sys.stdout.write(f"\rAnalyzed batch {b+1}/{num_batches}")
        sys.stdout.flush()
        
    print("\nSegmentation phase...")
    
    # Resolve overlapping windows into clean segments
    fps = 100  # 0.01s resolution
    timeline_length = int(total_duration * fps) + 1
    labels_timeline = np.zeros(timeline_length, dtype=int)
    
    for p in predictions:
        start_idx = int(p['start'] * fps)
        end_idx = int(p['end'] * fps)
        
        # 0 = Speaking (base), 2 = Breath (wins overlaps)
        # Stamping the full 0.5s window leaves natural padding around each breath.
        if p['label'] == 'Breath':
            stamp_end_idx = min(end_idx, timeline_length)
            
            labels_timeline[start_idx:stamp_end_idx] = np.maximum(labels_timeline[start_idx:stamp_end_idx], 2)
            
    # Walk the timeline and merge runs into segments
    final_segments = []
    current_label = labels_timeline[0]
    segment_start = 0
    
    for i in range(1, timeline_length):
        if labels_timeline[i] != current_label:
            final_segments.append({
                'label_idx': current_label,
                'start': segment_start / fps,
                'end': i / fps
            })
            current_label = labels_timeline[i]
            segment_start = i
            
    final_segments.append({
        'label_idx': current_label,
        'start': segment_start / fps,
        'end': total_duration
    })
    
    # Clean up short stutter segments:
    # Speaking < 0.5s is probably a pause inside a breath
    # Breath < 0.2s is probably a false positive
    for seg in final_segments:
        duration = seg['end'] - seg['start']
        if seg['label_idx'] == 0 and duration < 0.5:
            seg['label_idx'] = 2 # Flip to Breath
        elif seg['label_idx'] == 2 and duration < 0.2:
            seg['label_idx'] = 0 # Flip to Speaking
            
    # Re-merge after flipping
    merged_segments = []
    for seg in final_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            if merged_segments[-1]['label_idx'] == seg['label_idx']:
                merged_segments[-1]['end'] = seg['end']
            else:
                merged_segments.append(seg)
                
    final_segments = merged_segments
    
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    out_b = os.path.join(output_dir, "Inference_Output", base_name, "Breath")
    
    os.makedirs(out_b, exist_ok=True)
    
    b_count = 0
    
    print("Exporting audio clips and Audacity labels...")
    label_file_path = os.path.join(output_dir, "Inference_Output", base_name, f"{base_name}_inference_labels.txt")
    with open(label_file_path, "w", encoding="utf-8") as lf:
        for seg in final_segments:
            start_time = seg['start']
            end_time = seg['end']
            
            if seg['label_idx'] == 2:
                lf.write(f"{start_time:.3f}\t{end_time:.3f}\tBreath\n")
                
                # Skip tiny glitches
                if end_time - start_time >= 0.2:
                    audio_clip = audio_full[start_time * 1000 : end_time * 1000]
                    b_count += 1
                    filename = os.path.join(out_b, f"{base_name}_Breath_{b_count:03d}.wav")
                    audio_clip.export(filename, format="wav")
                
    print(f"Done! Exported: {b_count} Breaths.")

    if '--no-gui' not in sys.argv:
        viewer_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waveform_viewer.py")
        if os.path.exists(viewer_script):
            print(f"Launching waveform viewer for {wav_file}...")
            import subprocess
            subprocess.Popen(
                [sys.executable, viewer_script, wav_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

if __name__ == "__main__":
    wav_file = None
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".wav"):
            wav_file = arg
            break
            
    if not wav_file:
        print("Usage: Drag and drop a .wav file onto this script.")
        print("Or run: python inference.py <audio.wav>")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'best_mel_cnn.pth')
    
    try:
        run_inference(wav_file, model_path, base_dir)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        input("\nPress Enter to exit...")
