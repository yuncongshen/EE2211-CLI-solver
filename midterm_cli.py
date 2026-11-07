"""Command line entry point: python midterm_cli.py --help."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from calculations import parse_matrix, inspect_array, solve_system, regress


def parser():
    root = argparse.ArgumentParser(description='EE2211 matrix and regression calculator')
    commands = root.add_subparsers(dest='command', required=True)
    from extras import add_commands
    add_commands(commands)
    array = commands.add_parser('array', help='Inspect a matrix and its inverses')
    array.add_argument('matrix', nargs='?', help='Quoted matrix, e.g. "1 2; 3 4"')
    array.add_argument('--file', help='Headerless CSV or text matrix')
    solve = commands.add_parser('solve', help='Solve Xw = y or wX = y')
    solve.add_argument('--equation', choices=['right', 'left'], default='right', help='right: Xw=y (default); left: wX=y')
    regression = commands.add_parser('regress', help='Fit a model and optionally predict')
    regression.add_argument('model', choices=['linear', 'polynomial', 'ridge', 'ridge-polynomial'])
    regression.add_argument('--degree', type=int, default=2)
    regression.add_argument('--lambda', dest='ridge_lambda', type=float, default=1.0)
    regression.add_argument('--bias', action=argparse.BooleanOptionalAction, default=True, help='Include intercept (default: yes)')
    regression.add_argument('--penalize-bias', action=argparse.BooleanOptionalAction, default=True, help='Ridge penalizes intercept by default, matching the source app')
    regression.add_argument('--form', choices=['auto', 'primal', 'dual'], default='auto')
    regression.add_argument('--target', choices=['numeric', 'sign', 'onehot'], default='numeric')
    for name in ('predict', 'y-test'):
        group = regression.add_mutually_exclusive_group()
        group.add_argument('--' + name)
        group.add_argument('--' + name + '-file')
    for command in (solve, regression):
        for name in ('x', 'y'):
            group = command.add_mutually_exclusive_group(required=True)
            group.add_argument('--' + name, help='Quoted matrix; semicolons separate rows')
            group.add_argument('--' + name + '-file', help='Headerless CSV or text matrix')
    for command in (array, solve, regression):
        command.add_argument('--precision', type=int, default=4, help='Decimal places, 0 to 15 (default: 4)')
        command.add_argument('--json', action='store_true', help='Print full machine-readable results')
    return root


def read_input(text, filename, mixed=False):
    if text is not None and filename is not None:
        raise ValueError('Provide either inline data or a file, not both.')
    if filename is not None:
        text = Path(filename).read_text(encoding='utf-8-sig')
    return None if text is None else parse_matrix(text, mixed=mixed)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def display(result, args):
    precision = args.precision
    def fmt(value):
        if isinstance(value, (float, int, np.number)):
            return f'{0 if abs(value) < 0.5 * 10 ** -precision else value:.{precision}f}'
        return str(value)
    def matrix(label, value):
        print(label)
        if value is None:
            print('  Not available')
        else:
            rows = [[fmt(cell) for cell in row] for row in np.atleast_2d(value)]
            widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
            for row in rows:
                print('  [ ' + '  '.join(cell.rjust(width) for cell, width in zip(row, widths)) + ' ]')
        print()
    def section(title):
        print(title)
        print('-' * max(40, len(title)))
    def table(label, headers, values):
        section(label)
        rows = [[fmt(cell) for cell in row] for row in np.atleast_2d(np.asarray(values, dtype=object))]
        widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
        print('  ' + '  '.join(header.rjust(width) for header, width in zip(headers, widths)))
        print('  ' + '  '.join('-' * width for width in widths))
        for row in rows:
            print('  ' + '  '.join(cell.rjust(width) for cell, width in zip(row, widths)))
        print()
    def metrics(data, label='Error summary', axis='Output'):
        keys = [key for key in ('sse', 'mse', 'rmse') if key in data]
        values = np.vstack([data[key].ravel() for key in keys]).T
        table(label, [axis] + [key.upper() for key in keys],
              [[str(i + 1), *row] for i, row in enumerate(values)])
    if args.command == 'array':
        section('Array inspection')
        matrix('Matrix A', result['matrix'])
        print(f"Shape         {result['shape'][0]} x {result['shape'][1]}")
        print(f"Rank          {result['rank']}")
        print('Determinant   ' + (fmt(result['determinant']) if result['determinant'] is not None else 'N/A (not square)'))
        print('Invertible    ' + ('Yes' if result['inverse'] is not None else 'No') + '\n')
        matrix('Transpose', result['transpose'])
        if result['inverse'] is not None:
            matrix('Inverse (also the left and right inverse)', result['inverse'])
        else:
            matrix('Left inverse', result['left_inverse'])
            matrix('Right inverse', result['right_inverse'])
    elif args.command == 'solve':
        section('Solving ' + result['equation'])
        print('System: ' + result['system_type'])
        print(f"Equations: {result['equations']}    Unknowns: {result['unknowns']}    Rank: {result['rank']}\n")
        matrix('X', result['x'])
        matrix('y', result['y'])
        print('Solution: ' + result['solution_type'])
        print('Result: ' + result['status'])
        if len(result['target_statuses']) > 1:
            axis = 'Target row' if args.equation == 'left' else 'Target column'
            for i, status in enumerate(result['target_statuses'], 1):
                print(f'  {axis} {i}: {status}')
        print('Inverse form: ' + result['inverse_type'])
        print('Weights: ' + result['weight_formula'])
        print('Computed using stable least squares; the inverse formula is mathematically equivalent.')
        print('Exactness is checked within floating-point tolerance.\n')
        matrix('w', result['w'])
        section('Verification')
        matrix('Predicted y', result['predicted'])
        matrix('Residual (predicted - actual)', result['residual'])
        metrics(result, axis='Target row' if args.equation == 'left' else 'Output')
    else:
        section(result['model'].replace('-', ' ').title() + ' regression')
        print(f"Samples: {result['samples']}    Features: {result['features']}    Intercept: {'Yes' if result['bias'] else 'No'}")
        print('Method: ' + result['method'])
        print('Inverse form: ' + result['inverse_type'])
        print(f"Design matrix A: {result['design'].shape[0]} x {result['design'].shape[1]}, rank {result['design_rank']}")
        print('Weights: ' + result['weight_formula'])
        print('A includes the selected bias and polynomial terms.')
        print(result['computation'] + '\n')
        names = ['Intercept' if name == '1' else name for name in result['feature_names']]
        table('Design matrix', result['feature_names'], result['design'])
        table('Coefficients', ['Term'] + [f'w{i+1}' for i in range(result['w'].shape[1])],
              [[name, *row] for name, row in zip(names, result['w'])])
        if result['w'].shape[1] == 1:
            terms = [f"({fmt(row[0])})" + ('' if name == '1' else f'*{name}') for name, row in zip(result['feature_names'], result['w'])]
            print('Fitted equation\n  y = ' + ' + '.join(terms) + '\n')
        for i in range(result['actual'].shape[1]):
            table(f'Training results - output {i+1}', ['Actual', 'Predicted', 'Residual'],
                  np.column_stack((result['actual'][:, i], result['predicted'][:, i], result['residual'][:, i])))
        metrics(result, 'Training errors')
        if result['classes'] is not None:
            matrix('Training classes', result['classes'])
        if 'test_predicted' in result:
            table('Test design matrix', result['feature_names'], result['test_design'])
            table('Predictions', [f'x{i+1}' for i in range(result['test_input'].shape[1])] +
                  [f'Predicted y{i+1}' for i in range(result['test_predicted'].shape[1])],
                  np.hstack((result['test_input'], result['test_predicted'])))
            if result['test_classes'] is not None:
                matrix('Predicted classes', result['test_classes'])
            if 'test_errors' in result:
                matrix('Test residual (predicted - actual)', result['test_errors']['residual'])
                metrics(result['test_errors'], 'Test errors')
        if result['labels'] is not None:
            print('Class order: ' + ', '.join(str(label) for label in result['labels']) + '\n')
        section('Calculation reference')
        print('Prediction = design matrix @ w')
        print('Residual = predicted - actual')
        print('SSE = sum(residual^2); MSE = SSE / samples; RMSE = sqrt(MSE)')
        if args.model in ('ridge', 'ridge-polynomial'):
            print('Ridge minimizes SSE + lambda * sum(penalized weights^2).')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from interactive import run_session
        return run_session(display)
    args = parser().parse_args(argv)
    try:
        if not 0 <= args.precision <= 15:
            raise ValueError('Precision must be between 0 and 15.')
        if args.command in ('pearson', 'gradient', 'impurity', 'tree', 'kmeans'):
            from extras import cli, report
            result = cli(args, read_input)
            if args.json:
                print(json.dumps(json_safe(result), indent=2, allow_nan=False))
            else:
                report(args.command, result, args.precision)
            return 0
        if args.command == 'array':
            a = read_input(args.matrix, args.file)
            if a is None:
                raise ValueError('Provide a matrix or --file.')
            result = inspect_array(a)
        else:
            x = read_input(args.x, args.x_file)
            y = read_input(args.y, args.y_file, mixed=args.command == 'regress' and args.target == 'onehot')
            if args.command == 'solve':
                result = solve_system(x, y, args.equation)
            else:
                result = regress(x, y, model=args.model, degree=args.degree, bias=args.bias,
                                 ridge_lambda=args.ridge_lambda, form=args.form,
                                 penalize_bias=args.penalize_bias, target=args.target,
                                 predict=read_input(args.predict, args.predict_file),
                                 y_test=read_input(args.y_test, args.y_test_file, mixed=args.target == 'onehot'))
        if args.json:
            print(json.dumps(json_safe(result), indent=2, allow_nan=False))
        else:
            display(result, args)
        return 0
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

