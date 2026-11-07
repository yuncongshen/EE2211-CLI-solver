"""Additional calculations and full terminal reports."""
import ast
import numpy as np
import sympy as sp
from reference_math.pearson_correlation import pearson_correlation_details
from reference_math.DecisionTreeUtils import (classification_impurity_summary, regression_node_mse,
    regression_threshold_summary, find_best_regression_split)
from reference_math.KMeansUtils import k_means


def expression(text):
    """Parse mathematical syntax without evaluating Python code."""
    functions = {name: getattr(sp, name) for name in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs')}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.sympify(node.value)
        if isinstance(node, ast.Name):
            return {'pi': sp.pi, 'E': sp.E}.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            operations = {ast.Add: lambda: a+b, ast.Sub: lambda: a-b, ast.Mult: lambda: a*b,
                          ast.Div: lambda: a/b, ast.Pow: lambda: a**b}
            if type(node.op) in operations:
                return operations[type(node.op)]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and len(node.args) == 1 and not node.keywords:
            return functions[node.func.id](visit(node.args[0]))
        raise ValueError('Use numbers, variables, + - * / ^, and sin/cos/tan/exp/log/sqrt/Abs(...).')
    try:
        result = visit(ast.parse(text.replace('^', '**'), mode='eval').body)
        if not result.free_symbols:
            raise ValueError('The cost must contain at least one variable.')
        return result
    except (SyntaxError, TypeError, ZeroDivisionError) as exc:
        raise ValueError('Invalid mathematical expression.') from exc


def gradient(cost, initial, rate=0.1, iterations=10, tolerance=None):
    expr = expression(cost)
    symbols = sorted(expr.free_symbols, key=str)
    values = np.asarray(initial, dtype=float).ravel()
    if len(values) != len(symbols):
        raise ValueError(f'Expected {len(symbols)} initial values in order: {", ".join(map(str, symbols))}.')
    if not np.isfinite(values).all() or not np.isfinite(rate) or rate < 0:
        raise ValueError('Initial values and learning rate must be finite; rate must be nonnegative.')
    if iterations < 1 or int(iterations) != iterations:
        raise ValueError('Iterations must be a positive integer.')
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Tolerance must be finite and nonnegative.')
    derivatives = [sp.diff(expr, symbol) for symbol in symbols]
    history = []
    stopped = 'Iteration limit reached'
    for step in range(iterations + 1):
        try:
            subs = dict(zip(symbols, values))
            cost_value = float(expr.evalf(subs=subs))
            grad = np.array([float(d.evalf(subs=subs)) for d in derivatives])
            if not np.isfinite(cost_value) or not np.isfinite(grad).all() or not np.isfinite(values).all():
                raise ValueError()
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Non-finite or non-real result at iteration {step}; check the domain or reduce the learning rate.') from exc
        history.append(dict(iteration=step, values=values.copy(), gradient=grad, cost=cost_value))
        if step and tolerance is not None:
            previous = history[-2]
            if np.max(np.abs(values-previous['values'])) <= tolerance or abs(cost_value-previous['cost']) <= tolerance:
                stopped = 'Tolerance reached (parameter change or cost change)'
                break
        values = values-rate*grad
    return dict(function=str(expr), variables=list(map(str, symbols)), derivatives=list(map(str, derivatives)),
                history=history, stop=stopped, final_parameters=history[-1]['values'].reshape(-1, 1))


def calculate(kind, **args):
    if kind == 'pearson':
        return pearson_correlation_details(args['x'], args['y'])
    if kind == 'gradient':
        return gradient(**args)
    if kind == 'impurity':
        counts = args['counts']
        if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts.sum(axis=1) <= 0):
            raise ValueError('Counts must be finite and nonnegative, with a positive total in every row.')
        measures = ['gini', 'entropy', 'misclassification'] if args['measure'] == 'all' else [args['measure']]
        result = dict(counts=counts, probabilities=counts/counts.sum(axis=1, keepdims=True))
        for measure in measures:
            sizes, impurity, weighted = classification_impurity_summary(counts, measure)
            result[measure] = dict(node_sizes=sizes, per_node=impurity, weighted=weighted)
        return result
    if kind == 'tree':
        x, y = args['x'], args['y']
        if min(x.shape) != 1 or min(y.shape) != 1 or x.size != y.size or x.size < 2:
            raise ValueError('Use x and y vectors of equal length, with at least two samples.')
        mean, mse = regression_node_mse(y)
        result = dict(root_mean=mean, root_mse=mse)
        if args.get('threshold') is not None:
            if not np.isfinite(args['threshold']):
                raise ValueError('Threshold must be finite.')
            result['chosen_split'] = regression_threshold_summary(x, y, args['threshold'])
        if args.get('best', True):
            result['best_split'], result['candidates'] = find_best_regression_split(x, y)
        return result
    if kind == 'kmeans':
        history = k_means(args['x'], args['centroids'], args.get('iterations'))
        return dict(history=history, final_centroids=history[-1]['centroids'],
                    clusters=(history[-1]['assignments']+1).reshape(-1, 1),
                    stop='Converged' if history[-1]['converged'] else 'Iteration limit reached',
                    convention='Clusters are numbered from 1; ties select the first centroid; empty clusters retain their centroid.')
    raise ValueError('Unknown calculation.')


def report(title, result, precision=4):
    print('\n' + title.upper() + '\n' + '='*60)
    def table(headers, rows):
        formatted = [[f'{v:.{precision}f}' if isinstance(v, (float, np.floating)) else str(v) for v in row] for row in rows]
        widths = [max(len(h), *(len(row[i]) for row in formatted)) for i, h in enumerate(headers)]
        print('  '.join(h.rjust(w) for h, w in zip(headers, widths)))
        print('  '.join('-'*w for w in widths))
        for row in formatted:
            print('  '.join(v.rjust(w) for v, w in zip(row, widths)))
    def show(label, value, indent=''):
        print(indent + label.replace('_', ' ') + ':')
        if isinstance(value, dict):
            for key, item in value.items():
                show(key, item, indent+'  ')
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for index, item in enumerate(value):
                show(f'Entry {index}', item, indent+'  ')
        elif isinstance(value, np.ndarray):
            print(np.array2string(value, precision=precision, suppress_small=True, threshold=np.inf, prefix=indent+'  '))
        elif isinstance(value, (float, np.floating)):
            print(indent + f'  {value:.{precision}f}')
        else:
            print(indent + '  ' + str(value))
    for key, value in result.items():
        if key == 'history' and title == 'gradient':
            print('Iteration history (gradient evaluated at the displayed parameters):')
            table(['Step', *result['variables'], *['dC/d'+v for v in result['variables']], 'Cost'],
                  [[item['iteration'], *item['values'], *item['gradient'], item['cost']] for item in value])
            continue
        if key == 'candidates':
            print('Candidate thresholds:')
            table(['Threshold', 'Left mean', 'Right mean', 'Left MSE', 'Right MSE', 'Weighted MSE'],
                  [[item[k] for k in ('threshold', 'left_mean', 'right_mean', 'left_mse', 'right_mse', 'overall_mse')] for item in value])
            continue
        if key == 'history' and title == 'kmeans':
            for item in value:
                print(f"\nIteration {item['iteration']} | SSE cost: {item['cost']:.{precision}f}")
                show('Centroids', item['centroids'])
                table(['Point', 'Cluster', *[f'Distance to {i+1}' for i in range(item['distances'].shape[1])]],
                      [[i+1, int(cluster)+1, *item['distances'][i]] for i, cluster in enumerate(item['assignments'])])
            continue
        show(key, value)
    print()


def add_commands(commands):
    parsers = {}
    for name in ('pearson', 'gradient', 'impurity', 'tree', 'kmeans'):
        p = commands.add_parser(name, help=f'{name.title()} calculation')
        p.add_argument('--json', action='store_true')
        p.add_argument('--precision', type=int, default=4)
        parsers[name] = p
    for name, inputs in [('pearson', ['x', 'y']), ('gradient', ['initial']), ('impurity', ['counts']), ('tree', ['x', 'y']), ('kmeans', ['x', 'centroids'])]:
        for field in inputs:
            group = parsers[name].add_mutually_exclusive_group(required=True)
            group.add_argument('--'+field)
            group.add_argument('--'+field+'-file')
    p = parsers['gradient']
    p.add_argument('--cost', required=True)
    p.add_argument('--rate', type=float, default=0.1)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--tolerance', type=float)
    parsers['impurity'].add_argument('--measure', choices=['all', 'gini', 'entropy', 'misclassification'], default='all')
    parsers['tree'].add_argument('--threshold', type=float)
    parsers['tree'].add_argument('--best', action=__import__('argparse').BooleanOptionalAction, default=True)
    parsers['kmeans'].add_argument('--iterations', type=int)


def cli(args, read_input):
    fields = {'pearson': ['x', 'y'], 'gradient': ['initial'], 'impurity': ['counts'], 'tree': ['x', 'y'], 'kmeans': ['x', 'centroids']}[args.command]
    kwargs = {field: read_input(getattr(args, field), getattr(args, field+'_file')) for field in fields}
    settings = {'pearson': [], 'gradient': ['cost', 'rate', 'iterations', 'tolerance'], 'impurity': ['measure'], 'tree': ['threshold', 'best'], 'kmeans': ['iterations']}[args.command]
    kwargs.update({field: getattr(args, field) for field in settings})
    return calculate(args.command, **kwargs)
