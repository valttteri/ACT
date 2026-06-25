import torch
import os
import glob
from dotenv import load_dotenv
import pynvml


def check_cuda():
    print("\n###### Check cuda status ######\n")
    try:
        is_cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        curr_devices = torch.cuda.current_device()
        device_id = torch.cuda.device(0)
        device_name = torch.cuda.get_device_name(0)
    except RuntimeError:
        print("No GPU available")
        return

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
        print("HF_HOME was set incorrectly as", hf_dir)
        return False
    
    print("HF_HOME correct", hf_dir)
    return True

def check_gpu_usage():
    print(f"Current GPU usage: {torch.cuda.memory.memory_reserved()/1e9:.2f} GB")

def check_pickle_order():
    #llama_7b_head_wise_chunk_2
    chunks = sorted(glob.glob(f'ACT/activations/llama_7b_head_wise_chunk_*.pkl'))
    print("chunks", chunks)

def check_nvml():
    try:
        print("Initializing pynvml")
        pynvml.nvmlInit()
        print(f"Driver Version: {pynvml.nvmlSystemGetDriverVersion()}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    #check_gpu_usage(msg="Checking")
    check_nvml()