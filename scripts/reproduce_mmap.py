
import os
import time
import numpy as np
from mmap_frame_buffer import MmapFrameWriter, MmapFrameReader, get_mmap_path

def test_mmap_reuse():
    task_id = "test_reuse_task"
    frames = 100
    shape = (480, 848, 3) # Typical size
    
    print("--- First Initialization ---")
    start = time.time()
    writer1 = MmapFrameWriter(task_id, frames, shape)
    print(f"Init 1 took: {time.time() - start:.4f}s")
    
    # Simulate writing some frames
    writer1.write_frame(0, np.zeros(shape, dtype=np.uint8))
    writer1.close()
    
    # Check file mtime
    path = get_mmap_path(task_id)
    mtime1 = os.path.getmtime(path)
    
    print("\n--- Second Initialization (Same params) ---")
    start = time.time()
    writer2 = MmapFrameWriter(task_id, frames, shape)
    print(f"Init 2 took: {time.time() - start:.4f}s")
    
    mtime2 = os.path.getmtime(path)
    
    if mtime2 > mtime1:
        print("\n[FAIL] File was recreated (mtime changed)")
    else:
        print("\n[PASS] File was reused (mtime unchanged)")
        
    writer2.close()
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    test_mmap_reuse()
