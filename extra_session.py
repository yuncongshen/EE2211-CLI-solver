"""Guided workflows for the additional reference features."""
import json
from pathlib import Path
import numpy as np
from extras import calculate, expression, report

MENU = '4 Pearson | 5 Gradient descent | 6 Tree impurity | 7 Regression tree | 8 K-means | 9 Cache'
KINDS = {'4': 'pearson', '5': 'gradient', '6': 'impurity', '7': 'tree', '8': 'kmeans'}


def scalar(session, label, integer=False):
    while True:
        raw = session.ask(f'{label}: finite {"positive integer" if integer else "number"} | Enter=none | back=cancel\n> ')
        if not raw:
            return None
        try:
            value = int(raw) if integer else float(raw)
            if not np.isfinite(value) or (integer and value < 1):
                raise ValueError()
            return value
        except ValueError:
            print('Invalid value; try again.')


def guided(session, kind):
    s = session
    kwargs = {}
    if kind == 'pearson':
        print('Rows are observations; columns are variables. Matching row vectors are transposed automatically.')
        kwargs['x'] = s.matrix_prompt('X')
        def valid(y):
            calculate(kind, x=kwargs['x'], y=y)
        kwargs['y'] = s.matrix_prompt('y', validate=valid)
    elif kind == 'gradient':
        while True:
            cost = s.ask('Cost: variables, + - * / ^; sin/cos/tan/exp/log/sqrt/Abs; pi/E\nExample: x^2 + x*y^2 | back=cancel\n> ')
            try:
                expr = expression(cost)
                break
            except ValueError as exc:
                print(f'Error: {exc}')
        variables = sorted(map(str, expr.free_symbols))
        print('Initial value order: ' + ', '.join(variables))
        def valid(initial):
            if initial.size != len(variables):
                raise ValueError(f'Enter {len(variables)} initial values.')
        kwargs = dict(cost=cost, initial=s.matrix_prompt('initial', validate=valid),
                      rate=s.number('Learning rate', 0.1), iterations=s.number('Iterations', 10, integer=True))
        if s.choice('Use stopping tolerance?', ['yes', 'no'], 'no') == 'yes':
            kwargs['tolerance'] = s.number('Tolerance (parameter or cost change)', 0.000001)
    elif kind == 'impurity':
        print('Each row is a node; each column is a class count. Example: 3 4; 6 0')
        kwargs['measure'] = s.choice('Impurity measure', ['all', 'gini', 'entropy', 'misclassification'], 'all')
        def valid(counts):
            calculate(kind, counts=counts, measure=kwargs['measure'])
        kwargs['counts'] = s.matrix_prompt('counts', validate=valid)
    elif kind == 'tree':
        def vector(x):
            if min(x.shape) != 1 or x.size < 2:
                raise ValueError('Enter a row or column vector with at least two samples.')
        kwargs['x'] = s.matrix_prompt('x', validate=vector)
        def valid(y):
            vector(y)
            if y.size != kwargs['x'].size:
                raise ValueError('x and y must have the same number of samples.')
        kwargs['y'] = s.matrix_prompt('y', validate=valid)
        while True:
            threshold = scalar(s, 'Optional threshold (left: x <= threshold, right: x > threshold)')
            try:
                if threshold is not None:
                    calculate(kind, **kwargs, threshold=threshold, best=False)
                break
            except ValueError as exc:
                print(f'Error: {exc}')
        kwargs['threshold'] = threshold
        if np.unique(kwargs['x']).size < 2:
            print('All x values are equal; no valid split exists. Showing root statistics.')
            kwargs['best'] = False
        else:
            kwargs['best'] = s.choice('Find best split and show every candidate?', ['yes', 'no'], 'yes') == 'yes'
    elif kind == 'kmeans':
        print('Each row is a point. Enter initial centroid rows with the same number of feature columns.')
        kwargs['x'] = s.matrix_prompt('X')
        def valid(c):
            if c.shape[1] != kwargs['x'].shape[1]:
                raise ValueError('Centroids must have the same number of columns as X.')
        kwargs['centroids'] = s.matrix_prompt('initial_centroids', validate=valid)
        kwargs['iterations'] = scalar(s, 'Maximum iterations (blank runs until convergence)', integer=True)
    result = calculate(kind, **kwargs)
    report(kind, result)
    cache_result(s, kind, result)


def cache_result(session, prefix, result):
    for name, value in result.items():
        if isinstance(value, np.ndarray):
            session.save(prefix+'_'+name, value)
    print('Outputs saved under ' + prefix + '_ names. Use show to list them.')


def cache_menu(s):
    action = s.choice('Cache action', ['list', 'show', 'delete', 'clear', 'save', 'load'], 'list')
    if action == 'list':
        s.execute('show')
        return
    if action in ('show', 'delete'):
        print('Names: ' + ', '.join(s.matrices))
        name = s.ask('Saved matrix name | back=cancel\n> ')
        if name not in s.matrices:
            raise ValueError('No matrix with that name.')
        if action == 'show':
            s.execute('show '+name)
        else:
            del s.matrices[name]
            print('Deleted '+name)
    elif action == 'clear':
        if s.choice('Clear all cached matrices?', ['yes', 'no'], 'no') == 'yes':
            s.matrices.clear()
    else:
        path = Path(s.ask('JSON session file path (e.g. session.json) | back=cancel\n> '))
        if action == 'save':
            if path.exists() and s.choice('Replace existing file?', ['yes', 'no'], 'no') != 'yes':
                return
            path.write_text(json.dumps({key: value.tolist() for key, value in s.matrices.items()}, indent=2), encoding='utf-8')
            print('Saved matrices to '+str(path))
        else:
            from calculations import parse_matrix
            import re
            content = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(content, dict):
                raise ValueError('Session file must contain a dictionary of named matrices.')
            parsed = {}
            for name, value in content.items():
                if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
                    raise ValueError('Invalid matrix name in file.')
                parsed[name] = parse_matrix(repr(value), mixed=True)
            if set(parsed) & set(s.matrices) and s.choice('Replace matching saved names?', ['yes', 'no'], 'no') != 'yes':
                return
            for name, value in parsed.items():
                s.save(name, value)
