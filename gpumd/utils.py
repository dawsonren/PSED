import subprocess

def get_gpu_names():
    """
    Returns a list of GPU name strings, e.g. ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 SXM5 80GB"].
    Returns an empty list if no GPUs are found or nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return names
    except (FileNotFoundError, subprocess.CalledProcessError):
        # nvidia-smi not found, or command failed → probably a login node or non-GPU node
        return []


def detect_a100_h100():
    """
    Returns a dict with:
      - has_gpu_node: True if any NVIDIA GPU present
      - has_a100: True if any GPU name contains 'A100'
      - has_h100: True if any GPU name contains 'H100'
    """
    names = get_gpu_names()
    has_gpu_node = len(names) > 0

    has_a100 = any("a100" in name.lower() for name in names)
    has_h100 = any("h100" in name.lower() for name in names)

    return {
        "has_gpu_node": has_gpu_node,
        "has_a100": has_a100,
        "has_h100": has_h100,
        "gpu_names": names,
    }


if __name__ == "__main__":
    info = detect_a100_h100()
    HAS_GPU_NODE = info["has_gpu_node"]
    HAS_A100 = info["has_a100"]
    HAS_H100 = info["has_h100"]

    print("GPU node:", HAS_GPU_NODE)
    print("A100 present:", HAS_A100)
    print("H100 present:", HAS_H100)
    print("GPU names:", info["gpu_names"])