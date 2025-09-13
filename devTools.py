import inspect
import time


def get_current_line_number():
    frame = inspect.currentframe().f_back
    return frame.f_lineno


def run_function_with_timing(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    runtime = end_time - start_time
    return [result, runtime],
