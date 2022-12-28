import functools
import logging
import time

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()


def cut_unfinished_sentences(generation_func):
    @functools.wraps(generation_func)
    def wrapper(*args, **kwargs):
        text = generation_func(*args, **kwargs)
        if text[-1] not in '.!?':
            text = '.'.join(text.split('.')[:-1]) + '.'
        return text
    return wrapper


def log_generative_funcs(generation_func):
    @functools.wraps(generation_func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing function {generation_func.__name__}...")
        start_time = time.time()
        text = generation_func(*args, **kwargs)
        final_time = time.time() - start_time
        logger.info(f"Execution of function {generation_func.__name__} took {final_time:.2f}s.")
        return text
    return wrapper
