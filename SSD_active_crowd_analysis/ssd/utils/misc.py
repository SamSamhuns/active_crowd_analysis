import errno
import os


def str2bool(s):
    return s.lower() in ('true', '1')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def reset_range(old_max, old_min, new_max, new_min, arr):
    old_range = old_max - old_min
    print(old_range, 'old range')
    if old_range == 0:
        new_val = arr
        new_val[:] = new_min
    else:
        new_range = new_max - new_min
        new_val = (((arr - old_min) * new_range) / old_range) + new_min
    print(new_val, 'new_Val', arr)
    return new_val
