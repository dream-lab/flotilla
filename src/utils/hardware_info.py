import subprocess


def get_hardware_info():
    hardware_info = dict()
    try:
        output = subprocess.check_output(
            [
                "lscpu",
            ]
        )
    except Exception:
        output = None

    if output:
        output = output.decode("utf-8").strip().strip().split("\n")
        for line in output:
            line = line.split(":")
            if "Architecture" in line[0]:
                hardware_info["arch"] = line[1].strip()
            elif "CPU(s)" == line[0]:
                hardware_info["cpu_core_count"] = line[1].strip()
            elif "Model name" in line[0]:
                hardware_info["model_name"] = line[1].strip()
    else:
        hardware_info = {"arch": None, "cpu_core_count": None, "model_name": None}

    try:
        from torch.cuda import is_available

        hardware_info["cuda_available"] = True if is_available() else False
    except ModuleNotFoundError:
        hardware_info["cuda_available"] = False
    return hardware_info


if __name__ == "__main__":
    print(get_hardware_info())
