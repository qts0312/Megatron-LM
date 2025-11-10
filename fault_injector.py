"""
A class-based fault injector for monkey-patching functions
for distributed training frameworks, e.g., Megatron-LM.
"""

import torch
from megatron.core import mpu


class FaultInjector:
    """
    A generic fault injector that monkey-patches a target function to inject
    faults periodically.
    """
    def __init__(self, target_module,
                 target_function: str,
                 start_call: int = 0,
                 interval: int = 1,
                 max_injections: int = -1,
                 target_tp_rank: int = 0,
                 fault_type: str = 'nan'):
        self.target_module = target_module
        self.target_function = target_function
        self.start_call = start_call
        self.interval = interval
        self.max_injections = max_injections
        self.target_tp_rank = target_tp_rank
        self.fault_type = fault_type

        if self.error_type not in ['nan', 'inf', 'exception']:
            raise ValueError("error_type must be one of 'nan', 'inf', or 'exception'")
        if self.margin <= 0:
            raise ValueError("margin must be a positive integer.")
        if self.start_call < 0:
            raise ValueError("start_call must be a non-negative integer.")
        if self.max_injections < -1:
            raise ValueError("max_injections must be -1 or a non-negative integer.")
        
        self._original_function = None
        self._is_active = False
        self.reset_state()

    def reset_state(self):
        self._call_count = 0
        self._injection_count = 0

    def _should_inject(self):
        if self.max_injections != -1 and self._injections_count >= self.max_injections:
            return False
            
        if self._call_count < self.start_call:
            return False
            
        if (self._call_count - self.start_call) % self.interval == 0:
            return True
            
        return False
    
    def _patched_function(self, *args, **kwargs):
        if not self._is_active:
            return self._original_function(*args, **kwargs)

        try:
            if self._should_inject():
                current_tp_rank = mpu.get_tensor_model_parallel_rank()
                
                if current_tp_rank == self.target_tp_rank:
                    self._injections_count += 1
                    rank = torch.distributed.get_rank()
                    print(
                        f"[FaultInjector] Injecting {self.fault_type} to {self.target_module.__name__}.{self.target_function} on TP rank {current_tp_rank}, "
                        f"call count: {self._call_count}, injection count: {self._injection_count}",
                        flush=True
                    )
                    
                    # Assume tensor is the first argument
                    tensor = args[0]

                    if self.fault_type == 'nan':
                        tensor.data.fill_(float('nan'))
                    elif self.fault_type == 'inf':
                        tensor.data.fill_(float('inf'))
                    elif self.fault_type == 'exception':
                        # The exception will prevent the original function from being called
                        raise RuntimeError(f"Exception injected by FaultInjector")

            return self._original_function(*args, **kwargs)
        finally:
            self._call_count += 1

    def apply(self):
        if self._is_active:
            print("[FaultInjector]: Patch is already active.", flush=True)
            return

        if not hasattr(self.target_module, self.target_function):
            raise AttributeError(f"Module {self.target_module.__name__} has no function '{self.target_function}'")

        self.reset_state()
        self._original_function = getattr(self.target_module, self.target_function)
        setattr(self.target_module, self.target_function, self._patched_function)
        
        self._is_active = True
        limit_str = "unlimited" if self.max_injections == -1 else str(self.max_injections)
        print(f"[FaultInjector]: Patch applied to {self.target_module.__name__}.{self.target_function}\n"
              f"           Will inject up to {limit_str} times, starting at call "
              f"{self.start_call} with an interval of {self.interval} on TP rank {self.target_tp_rank}.",
              flush=True)
        
    def remove(self):
        if not self._is_active:
            return

        if self._original_function:
            setattr(self.target_module, self.target_function, self._original_function)
        
        self._original_function = None
        self._is_active = False
        print("[FaultInjector]: Patch removed.", flush=True)

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
