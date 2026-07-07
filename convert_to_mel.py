import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import concurrent.futures
import threading

def create_mel_spectrograms(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # AMD ROCm on Linux uses the 'cuda' backend in PyTorch.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    
    # Load the colormap once; matches the old librosa output colors
    cmap = plt.get_cmap('magma')
    
    # Transform cache + lock, shared across threads
    mel_transforms = {}
    amp_transforms = {}
    transform_lock = threading.Lock()
    
    tasks = []
    for root, dirs, files in os.walk(input_dir):
        # Skip the output directory
        if output_dir in root:
            continue
            
        for file in files:
            if Path(file).suffix.lower() in extensions:
                audio_path = os.path.join(root, file)
                
                # Mirror the input folder structure
                rel_path = os.path.relpath(root, input_dir)
                if rel_path == '.':
                    save_dir = output_dir
                else:
                    save_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(save_dir, exist_ok=True)
                
                filename = os.path.splitext(file)[0] + '.png'
                output_path = os.path.join(save_dir, filename)
                tasks.append((audio_path, output_path))
                
    if not tasks:
        print("No audio files were found to process.")
        return
        
    print(f"Found {len(tasks)} files to convert...")
    processed_count = [0]
    print_lock = threading.Lock()
    
    def process_single_file(task):
        audio_path, output_path = task
        try:
            # Disk I/O is slow; the thread pool hides the wait
            waveform, sr = torchaudio.load(audio_path)
            
            waveform = waveform.to(device)
            
            # Reuse cached transforms (thread-safe)
            with transform_lock:
                if sr not in mel_transforms:
                    mel_transforms[sr] = T.MelSpectrogram(
                        sample_rate=sr,
                        n_fft=2048,
                        hop_length=512,
                        n_mels=128,
                        f_max=8000.0
                    ).to(device)
                    amp_transforms[sr] = T.AmplitudeToDB().to(device)
                
            mel_spec = mel_transforms[sr](waveform)
            mel_spec_db = amp_transforms[sr](mel_spec)
            
            spec = mel_spec_db[0].cpu().numpy()
            
            # Normalize dB to 0..1
            spec_min = spec.min()
            spec_max = spec.max()
            if spec_max - spec_min > 0:
                spec_normalized = (spec - spec_min) / (spec_max - spec_min)
            else:
                spec_normalized = np.zeros_like(spec)
                
            # Flip so low frequencies are at the bottom
            spec_normalized = np.flipud(spec_normalized)
            
            spec_rgba = cmap(spec_normalized) 
            spec_rgb = (spec_rgba[:, :, :3] * 255).astype(np.uint8)
            
            image = Image.fromarray(spec_rgb, mode='RGB')
            
            image = image.resize((224, 224), Image.LANCZOS)
            
            image.save(output_path)
            
            with print_lock:
                processed_count[0] += 1
                print(f"\rProcessing... {processed_count[0]}/{len(tasks)} files", end="", flush=True)
                
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

    # Thread pool keeps the GPU fed while files load
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(process_single_file, tasks))
    print(f"\nCompleted {processed_count[0]} files!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_directory = None
    
    # Folder dropped onto the script
    if len(sys.argv) > 1:
        potential_dir = sys.argv[1]
        if os.path.isdir(potential_dir):
            input_directory = potential_dir
            print(f"Using dragged and dropped directory: {input_directory}")
        else:
            print(f"Error: Drag-and-dropped item is not a directory.")
            input("\nPress Enter to exit...")
            sys.exit(1)
    else:
        # Default: the Input folder next to the script
        input_directory = os.path.join(base_dir, "Input")
        if not os.path.exists(input_directory):
            print(f"Warning: directory '{input_directory}' does not exist.")
            print(f"Falling back to looking for audio in '{base_dir}' directly.")
            input_directory = base_dir
            
    output_directory = os.path.join(base_dir, "Training-Images")
    
    print(f"\n--- Starting GPU-Accelerated Mel-Spectrogram Generation ---")
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}\n")
    
    try:
        create_mel_spectrograms(input_directory, output_directory)
        print("Successfully finished processing!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    input("\nPress Enter to exit...")
