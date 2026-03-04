import torch



class LayeredProfiler:
    def __init__(self, print_memory=False):
        self.profiling_time_dict = {}
        self.print_memory = print_memory
        self.min_free_mem = 1024*1024*1024*1024


    def add_profiling_time(self, item, time, device: str = None):

        if not item in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

        self.profiling_time_dict[item].append(time)

        if self.print_memory:
            free_mem = None
            if device is not None and device.startswith("xpu"):
                try:
                    import intel_extension_for_pytorch as ipex  # noqa: F401
                    device_idx = int(device.split(":")[1]) if ":" in device else 0
                    props = torch.xpu.get_device_properties(device_idx)
                    allocated = torch.xpu.memory_allocated(device)
                    free_mem = props.total_memory - allocated
                except Exception:
                    free_mem = None
            elif device is not None and device.startswith("cuda"):
                try:
                    free_mem = torch.cuda.mem_get_info()[0]
                except Exception:
                    free_mem = None
            # CPU or device=None: skip profiling output (no meaningful VRAM info)

            if free_mem is not None:
                self.min_free_mem = min(self.min_free_mem, free_mem)
                print(f"free vmem @{item}: {free_mem/1024/1024/1024:.02f}GB, min free: {self.min_free_mem/1024/1024/1024:.02f}GB")

    def clear_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            self.profiling_time_dict[item] = []

    def print_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            print(f"total time for {item}: {sum(self.profiling_time_dict[item])}")
