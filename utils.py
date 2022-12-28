import functools


def cut_unfinished_sentences(generation_func):
    @functools.wraps(generation_func)
    def wrapper(*args, **kwargs):
        text = generation_func(*args, **kwargs)
        if text[-1] not in '.!?':
            text = '.'.join(text.split('.')[:-1]) + '.'
        return text
    return wrapper
