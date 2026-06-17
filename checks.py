import torch
import os
import glob
from dotenv import load_dotenv

def main():
    is_cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    curr_devices = torch.cuda.current_device()
    device_id = torch.cuda.device(0)
    device_name = torch.cuda.get_device_name(0)

    print("\n###### Check cuda status ######\n")
    print(f"Cuda is available: {is_cuda_available}")
    print(f"Device count: {device_count}")
    print(f"Current device: {curr_devices}")
    print(f"Device id: {device_id}")
    print(f"Device name: {device_name}\n")
    print("###############################")

def hf_home_is_correct():
    """Make sure HF_HOME is set correctly"""
    load_dotenv()
    target_dir = os.environ.get("PROJECT_DIR") + "/huggingface"
    hf_dir = os.environ.get("HF_HOME")
    
    if hf_dir != target_dir:
        print("HF_HOME set incorrectly as", hf_dir)
        return False
    
    print("HF_HOME correct", hf_dir)
    return True

def check_pickle_order():
    #llama_7b_head_wise_chunk_2
    chunks = sorted(glob.glob(f'ACT/activations/llama_7b_head_wise_chunk_*.pkl'))
    print("chunks", chunks)

if __name__ == "__main__":
    hf_home_is_correct()