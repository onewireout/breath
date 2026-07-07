import soundfile as sf
import numpy as np
import os

def edit_wav():
    wav_path = "Reviewed/Reviewed.wav"
    labels_path = "Reviewed/Reviewed.txt"
    output_path = "Reviewed/Edited.wav"

    if not os.path.exists(wav_path):
        print(f"Error: Could not find {wav_path}.")
        return
    if not os.path.exists(labels_path):
        print(f"Error: Could not find {labels_path}.")
        return

    print(f"Loading {wav_path}...")
    data, sr = sf.read(wav_path)

    # Convert 1-second clip (2s to 3s) into sample indices
    clip_start_idx = int(2.0 * sr)
    clip_end_idx = int(3.0 * sr)
    
    if len(data) < clip_end_idx:
        print("Error: The original wav file is shorter than 3 seconds.")
        return

    replacement_clip = data[clip_start_idx:clip_end_idx]
    clip_len = len(replacement_clip)

    print(f"Reading labels from {labels_path}...")
    with open(labels_path, 'r') as f:
        lines = f.readlines()

    print("Replacing labeled regions...")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) >= 2:
            try:
                start_time = float(parts[0])
                end_time = float(parts[1])
            except ValueError:
                continue

            start_idx = max(0, min(int(start_time * sr), len(data)))
            end_idx = max(0, min(int(end_time * sr), len(data)))
            
            duration_samps = end_idx - start_idx
            if duration_samps <= 0:
                continue
                
            reps = int(np.ceil(duration_samps / clip_len))
            
            # Tile differently for mono vs multi-channel
            if data.ndim > 1:
                tiled_clip = np.tile(replacement_clip, (reps, 1))
            else:
                tiled_clip = np.tile(replacement_clip, reps)
                
            final_replacement = tiled_clip[:duration_samps]
            
            data[start_idx:end_idx] = final_replacement

    print(f"Saving the edited audio to {output_path}...")
    info = sf.info(wav_path)
    sf.write(output_path, data, sr, subtype=info.subtype, format=info.format)
    print("Finished editing.")

if __name__ == "__main__":
    edit_wav()
