"""
Assistant Function for Co-RoPE：
- print_red_warning: print warning message in red
- calc_sim: calculate similarity between two tensors
- assert_similar: assert two tensors are similar
"""

def print_red_warning(message):
    # ANSI escape sequence: \033[31m set text to red, \033[0m reset color
    print(f"\033[31mWARNING: {message}\033[0m")

def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f'{name} all zero')
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim

def assert_similar(x, y, eps=1e-8, name="tensor", assert_=False, print_=True):
    ''' Correctness Standard:
        fwd: eps = 1e-5
        bwd (gradient): eps = 1e-4
    '''
    sim = calc_sim(x, y, name)
    diff = 1. - sim
    if not (0 <= diff <= eps):
        print_red_warning(f'{name} Error: {diff}')
        if assert_:
            assert False
    else:
        if print_:
            print(f'passed: {name} diff={diff}')