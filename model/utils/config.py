import os, subprocess, torch
from loguru import logger

def configure_nccl():
    """
    Configure multi-machine environment variables of NCCL(NVIDIA Collective Communications Library).
    """
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"

def get_gpu_nums():
    ngpus = torch.cuda.device_count()
    return ngpus
