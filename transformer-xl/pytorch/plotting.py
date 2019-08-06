import re
from pathlib import Path
from typing import Dict, List

import tabulate
from matplotlib import pyplot as plt


def parse_log(root: Path, name : str = '') -> Dict:
    with root.joinpath('log.txt').open() as f:
        result = {
            'path': root,
            'name': f'({name}) {root.name}' if name else root.name,
            'params': {},
            'train': [],
            'valid': [],
        }
        for line in f:
            param_match = re.match(r'^\s+- (\w+) : (.+)', line)
            if param_match:
                key, value = param_match.groups()
                result['params'][key] = value

            elif line.startswith('| epoch'):  # train log
                item = {field: float(
                    re.search(rf'\b{field}\s+(\d+\.?\d*)', line).groups()[0])
                        for field in ['step', 'epoch', 'lr', 'loss', 'ppl']}
                result['train'].append(item)

            elif line.startswith('| Eval'):
                item = {field: float(
                    re.search(rf'\b{field}\s+(\d+\.?\d*)', line).groups()[0])
                        for field in
                        ['Eval', 'step', 'valid loss', 'valid ppl']}
                result['valid'].append(item)

        return result


def plot_log(log: Dict,
             ymin: int = None, ymax: int = None,
             xmin: int = None, xmax: int = None):
    xlim_kw = {}
    if xmin is not None:
        xlim_kw['left'] = xmin
    if xmax is not None:
        xlim_kw['right'] = xmax
    if xlim_kw:
        plt.xlim(**xlim_kw)

    ylim_kw = {}
    if ymin is not None:
        ylim_kw['bottom'] = ymin
    if ymax is not None:
        ylim_kw['top'] = ymax
    if ylim_kw:
        plt.ylim(**ylim_kw)

    params = log['params']
    tokens_per_step = int(params['batch_size']) * int(params['tgt_len'])
    values = log['valid']
    xs = [item['step'] * tokens_per_step for item in values]
    ys = [item['valid ppl'] for item in values]
    plt.plot(xs, ys, label=log['name'])
    plt.legend()


def print_differing_params(logs: List[Dict]):
    assert len(logs) > 1
    log_params = [log['params'] for log in logs]
    all_keys = {k for params in log_params for k in params}
    values = {k: log_params[0].get(k) for k in all_keys}
    equal_keys = {k for k in all_keys
                  if all(p.get(k) == values[k] for p in log_params)}
    print_keys = sorted(
        all_keys - equal_keys -
        {'work_dir', 'restart_dir', 'eval_interval', 'log_interval',
         'n_nonemb_param'})
    headers = ['log'] + print_keys
    table = [[log['name']] + [log['params'].get(k) for k in print_keys]
             for log in logs]
    print(tabulate.tabulate(table, headers=headers))
