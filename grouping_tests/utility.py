import numpy as np
import base

def _method_require(**requirements):
    """
    Callable decorator to require that a class meets certain attribute or 
    property requirements.
    """
    def decorator(func):
        def wrapper(rng, *args, **kwargs):
            for key, value in requirements.items():
                if getattr(rng, key) != value:
                    raise ValueError(
                        f"The {func.__name__} method is only available "
                        f"for {rng.__class__.__name__} instances with {key}={value}."
                    )
            return func(rng, *args, **kwargs)
        return wrapper
    return decorator

def _prepare_data_array(data, name, ndim=1, dtype=None, copy=None):
    """
    Function for validating input data as a 1D scalar np.array using the given 
    name and optional dtype and copy arguments.
    """
    # Initialize data as a numpy array
    try:
        try:
            data = np.asarray(data, dtype=dtype, copy=copy)
        except TypeError: # numpy 1.X compatibility
            data = np.array(data, dtype=dtype, copy=False)
        assert data.ndim == ndim
    except AssertionError:
        if dtype is None:
            raise ValueError(
                f"Invalid input data for `{name}`. Must be a 1D array-like object."
            )
        else:
            raise ValueError(
                f"Invalid input data for `{name}`. Must be a 1D array-like object with dtype={dtype}. Provided array has shape={data.shape} and dtype={data.dtype}."
            )
    return data

def _represent_records(rng):
    """
    Create a string representation of a Rangel instance, displaying only the 
    first and last few records.
    """
    # If no ranges present, return self as a string
    if rng.num_events == 0:
        return str(rng)
    # Determine number of records to show
    display_max = rng._class_options['display_max']
    if rng.num_events > display_max:
        # Define head/skip/tail selections
        display_head = (display_max // 2) + (display_max % 2)
        display_tail = (display_max // 2)
        display_skip = rng.num_events - display_max
        # Define bool mask
        display_select = np.array(
            [True]  * display_head + 
            [False] * display_skip + 
            [True]  * display_tail)
    else:
        # Default head/skip/tail selections
        display_head = rng.num_events
        display_tail = display_skip = 0
        display_select = np.array([True] * rng.num_events)
    # Determine numbers of left and right digits to display
    ld = len(str(int(rng.arr[display_select].max())))
    rd = 3
    digits = ld + rd + 1
    # Create formatter
    records = []
    closed = rng.closed
    if rng.groups is not None:
        max_len = max([len(x) for x in rng.groups])
        groups = np.array([f'group({x[:20]: >{min(max_len, 20)}}) ' for x in rng.groups])
    else:
        groups = np.full(rng.num_events, '')
    # Define record string template and select features to display
    if rng.is_point:
        display_features = {
            'index': rng.index[display_select],
            'groups': groups[display_select],
            'locs': rng.locs[display_select],
            'modified_edges': rng.modified_edges[display_select],
        }
        record_template = '{index}, {groups}@ {locs: >{digits}.{rd}f}'
    elif rng.is_linear:
        display_features = {
            'index': rng.index[display_select],
            'groups': groups[display_select],
            'begs': rng.begs[display_select],
            'ends': rng.ends[display_select],
            'modified_edges': rng.modified_edges[display_select],
        }
        record_template = '{index}, {groups}{lb}{begs: >{digits}.{rd}f}, {ends: >{digits}.{rd}f}{rb}'
        if rng.is_located:
            display_features['locs'] = rng.locs[display_select]
            record_template += ' @ {locs: >{digits}.{rd}f}'

    # Iterate over selected features and create strings
    for i in range(sum(display_select)):
        params = {k: v[i] for k, v in display_features.items()}
        record = record_template.format(
            lb='[' if (closed in ['left','left_mod','both']) or params['modified_edges'] else '(',
            rb=']' if (closed in ['right','right_mod','both']) or params['modified_edges'] else ')',
            digits=digits, rd=rd, **params)
        records.append(record)

    # Create skipped record label if required
    if display_skip > 0:
        # Label skipped records
        spacer_label = '{:,.0f} records'.format(display_skip)
        # Format label
        spaces = max(ld*2 + rd*2 + 6 - len(spacer_label), 6)
        spacer = '.' * (spaces // 2) + spacer_label + \
            '.' * (spaces // 2 + spaces % 2)
        records = \
            records[:display_head] + [spacer] + records[-display_tail:]
    # Format full text string and return
    return '\n'.join(records) + '\n' + str(rng)
