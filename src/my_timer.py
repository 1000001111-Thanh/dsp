# timer.py
import time

class Timer:
    def __init__(self):
        self._start_stack = []  # Stack to track nested timers

    def tic(self):
        """Start a new timer (pushes start time onto the stack)"""
        self._start_stack.append(time.time())

    def toc(self, mask_as=None):
        """Stop the most recent timer and print elapsed time"""
        if not self._start_stack:
            raise RuntimeError("No timer started. Call tic() first.")
        elapsed_time = time.time() - self._start_stack.pop()
        print(f"Elapsed time {mask_as}: {elapsed_time:.4f} seconds")
        return elapsed_time

# Create a default instance for convenience
timer = Timer()
tic = timer.tic
toc = timer.toc