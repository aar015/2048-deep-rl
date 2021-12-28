"""Implement memory testing tools."""
from functools import wraps
from gc import collect
from tracemalloc import take_snapshot, Filter
from typing import List


def alloc_in_call(
    tag: str, filters: List[Filter] = [],
    key_type: str = 'filename', traceback_len: int = 5
):
    """Announce memory allocated in function."""
    def factory(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            snapshot1 = take_snapshot()
            result = func(*args, **kwargs)
            collect()
            snapshot2 = take_snapshot()
            print(tag)
            stats = snapshot2.filter_traces(filters).\
                compare_to(snapshot1.filter_traces(filters), key_type)
            for stat in stats[:10]:
                new_kb = stat.size_diff / 1024
                total_kb = stat.size / 1024
                print(f'  {new_kb} new KiB, {total_kb} total KiB')
                for line in stat.traceback.format()[:2 * traceback_len]:
                    print('    ' + line)
            return result
        return wrapper
    return factory


def alloc_since_call(
    tag: str, filters: List[Filter] = [],
    key_type: str = 'filename', traceback_len: int = 5
):
    """Announce memory allocated since last time function was called."""
    def factory(func):
        prev_snapshot = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            nonlocal prev_snapshot
            if prev_snapshot is None:
                prev_snapshot = take_snapshot()
                return result
            print(tag)
            snapshot = take_snapshot()
            stats = snapshot.filter_traces(filters).\
                compare_to(prev_snapshot.filter_traces(filters), key_type)
            for stat in stats[:10]:
                new_kb = stat.size_diff / 1024
                total_kb = stat.size / 1024
                print(f'\t{new_kb} new KiB, {total_kb} total KiB')
                for line in stat.traceback.format()[:2 * traceback_len]:
                    print('\t\t' + line)
            prev_snapshot = snapshot
            return result
        return wrapper
    return factory
