import os
import sys
import glob
import time
import argparse
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def compress_image(task):
    path, target_width, target_height = task
    try:
        with Image.open(path) as img:
            # Check if already small to avoid redundant work
            if img.size == (target_width, target_height) and img.mode == "L":
                return True, 0, 0
                
            orig_size = os.path.getsize(path)
            # Resize image
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            # Convert to 8-bit grayscale
            img_gray = img_resized.convert("L")
            # Overwrite in-place
            img_gray.save(path, format="PNG", optimize=True)
            
            new_size = os.path.getsize(path)
            return True, orig_size, new_size
    except Exception as e:
        return False, str(e), path

def main():
    parser = argparse.ArgumentParser(description="Compress OCR training images to target training dimensions and grayscale.")
    parser.add_argument("--dir", default="generated/training_1m", help="Directory containing the images")
    parser.add_argument("--width", type=int, default=256, help="Target image width")
    parser.add_argument("--height", type=int, default=48, help="Target image height")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel processes")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning directory: {args.dir}...")
    files = glob.glob(os.path.join(args.dir, "img_*.png"))
    total_files = len(files)
    
    if total_files == 0:
        print(f"No PNG images found matching 'img_*.png' in {args.dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {total_files:,} images.")
    
    # Calculate initial size
    print("Calculating initial size...")
    initial_total_bytes = 0
    # To be fast, let's use os.path.getsize directly in a loop
    for f in tqdm(files, desc="Summing file sizes"):
        initial_total_bytes += os.path.getsize(f)
        
    print(f"Initial dataset size: {initial_total_bytes / (1024*1024*1024):.2f} GB")
    print(f"Starting in-place compression to {args.width}x{args.height} Grayscale using {args.workers} workers...")

    # Build tasks
    tasks = [(f, args.width, args.height) for f in files]

    success_count = 0
    error_count = 0
    already_compressed = 0
    final_total_bytes = 0
    saved_bytes = 0

    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compress_image, task): task[0] for task in tasks}
        
        with tqdm(total=total_files, desc="Compressing") as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    success, val1, val2 = future.result()
                    if success:
                        if val1 == 0 and val2 == 0:
                            already_compressed += 1
                        else:
                            success_count += 1
                            saved_bytes += (val1 - val2)
                    else:
                        error_count += 1
                        print(f"\nError processing {val2}: {val1}", file=sys.stderr)
                except Exception as e:
                    error_count += 1
                    print(f"\nException processing {path}: {e}", file=sys.stderr)
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\nCompression finished in {elapsed_time:.2f} seconds.")
    print(f"  Successfully processed: {success_count:,}")
    if already_compressed > 0:
        print(f"  Already compressed: {already_compressed:,}")
    if error_count > 0:
        print(f"  Errors: {error_count:,}")

    # Calculate final size
    print("\nCalculating final size...")
    for f in files:
        final_total_bytes += os.path.getsize(f)

    initial_gb = initial_total_bytes / (1024**3)
    final_gb = final_total_bytes / (1024**3)
    reduction = (1 - final_total_bytes / initial_total_bytes) * 100 if initial_total_bytes > 0 else 0

    print("=" * 60)
    print(f"Initial Dataset Size: {initial_gb:.2f} GB")
    print(f"Final Dataset Size:   {final_gb:.2f} GB")
    print(f"Space Saved:          {(initial_total_bytes - final_total_bytes) / (1024**3):.2f} GB ({reduction:.1f}% reduction)")
    print(f"Average image size:   {final_total_bytes / total_files / 1024:.2f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()
