"""Interactive calculator with session-local matrices and model snapshots."""
import re
from types import SimpleNamespace
import numpy as np
from calculations import parse_matrix, inspect_array, solve_system, regress, encode_targets


class Back(Exception):
    pass


HELP = """
EE2211 Calculator
----------------------------------------
1  Array inspection
2  Solve an equation
3  Regression
4  Pearson correlation
5  Gradient descent
6  Decision-tree impurity
7  Regression-tree split
8  K-means clustering
9  Cache / save and load matrices
q  Quit

Shortcuts
  linear / polynomial / ridge / ridge-polynomial
                        Open full prompts for that model
  X = 0; 1; 2; 3        Save a matrix
  y = 1; 3; 5; 7
  set X                 Paste rows; finish with a blank line
  array X               Inspect a saved matrix
  solve X y             Solve Xw = y (add 'left' for wX = y)
  linear X y            Fit a line with an intercept
  ridge X y 0.1         Fit ridge with lambda=0.1
  poly X y 2            Fit a degree-2 polynomial
  ridgepoly X y 2 0.1   Polynomial degree, then lambda
  predict 4; 5          Predict with the last fitted model
  predict X_test        Predict using a saved matrix
  show X / show         View one matrix / list saved matrices
  edit X                Replace X by pasting rows
  transpose y           Transpose y in place
  history               View commands entered this session
  help / back           Show this menu

At any prompt, type back to cancel. Enter accepts shown defaults.
Matrices and the last model are kept until you quit.
"""


class Session:
    def __init__(self, display, reader=input):
        self.display = display
        self.reader = reader
        self.matrices = {}
        self.history = []
        self.last_fit = None

    def ask(self, prompt):
        value = self.reader(prompt).strip()
        if value.lower() == 'back':
            raise Back()
        return value

    def choice(self, prompt, choices, default):
        while True:
            print(f'\n{prompt}')
            print('  Options: ' + ' | '.join(choices))
            value = self.ask(f'  Enter = {default} | back = cancel\n> ').lower() or default
            if value in choices:
                return value
            print('Choose: ' + ', '.join(choices))

    def number(self, prompt, default, integer=False):
        while True:
            try:
                allowed = 'integer >= 1 (e.g. 1, 2, 3)' if integer else 'finite number >= 0 (e.g. 0, 0.1, 1)'
                print(f'\n{prompt}\n  Input: {allowed}')
                value = (int if integer else float)(self.ask(f'  Enter = {default} | back = cancel\n> ') or str(default))
                if not np.isfinite(value) or value < (1 if integer else 0):
                    raise ValueError()
                return value
            except ValueError:
                print('Enter a positive integer.' if integer else 'Enter a finite nonnegative number.')

    def save(self, name, matrix):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Matrix names must start with a letter and contain only letters, digits, or underscores.')
        self.matrices[name] = matrix.copy()
        print(f'Saved {name} ({matrix.shape[0]} x {matrix.shape[1]})')

    def numeric(self, matrix):
        try:
            value = np.asarray(matrix, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError('This operation requires numeric values.') from exc
        if not np.isfinite(value).all():
            raise ValueError('Values must be finite numbers.')
        return value

    def resolve(self, value, mixed=False):
        if value in self.matrices:
            matrix = self.matrices[value].copy()
            return matrix if mixed else self.numeric(matrix)
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', value):
            raise ValueError(f'No matrix named {value}. Use "set {value}" first.')
        return parse_matrix(value, mixed=mixed)

    def paste(self, mixed=False):
        print('Paste rows, then press Enter on an empty line to finish (back cancels).')
        rows = []
        while True:
            kind = 'numbers or class labels' if mixed else 'finite numbers'
            row = self.ask(f'Row: {kind}, spaces/commas separate columns | Enter = finish | back = cancel\n... ')
            if not row:
                break
            rows.append(row)
        return parse_matrix('\n'.join(rows), mixed=mixed)

    def matrix_prompt(self, name, mixed=False, validate=None, replace=False):
        while True:
            try:
                if replace:
                    matrix = self.paste(mixed)
                else:
                    hint = f'Enter=reuse {name}' if name in self.matrices else 'Enter=paste rows'
                    print(f'\n{name} input')
                    print('  Matrix: 1 2; 3 4 | column: 1; 2; 3 | nested array: [[1, 2], [3, 4]]')
                    if mixed:
                        print('  Class labels: low; high; low')
                    saved = ', '.join(f'{key} ({value.shape[0]} x {value.shape[1]})' for key, value in self.matrices.items())
                    print('  Saved names (case-sensitive): ' + (saved or 'none yet'))
                    text = self.ask(f'  {hint} | back=cancel\n> ')
                    matrix = self.resolve(text or name, mixed) if text or name in self.matrices else self.paste(mixed)
                if validate:
                    validate(matrix)
                self.save(name, matrix)
                return matrix
            except (ValueError, TypeError) as exc:
                print(f'Error: {exc} Try this input again.')

    def render(self, result, command, **options):
        self.display(result, SimpleNamespace(command=command, precision=4, **options))
        from extra_session import cache_result
        cache_result(self, command, result)
        print()

    def fit(self, x, y, predict=None, y_test=None, **options):
        result = regress(x, y, predict=predict, y_test=y_test, **options)
        self.last_fit = (x.copy(), y.copy(), options.copy())
        self.render(result, 'regress', model=options.get('model', 'linear'))

    def guided(self, operation, model=None):
        if operation == '1':
            self.render(inspect_array(self.matrix_prompt('A')), 'array')
            return
        if operation == '2':
            orientation = self.choice('Equation: right=Xw=y, left=wX=y', ['right', 'left'], 'right')
            x = self.matrix_prompt('X')
            def validate(y):
                axis = 0 if orientation == 'right' else 1
                if x.shape[axis] != y.shape[axis]:
                    raise ValueError(f'y needs {x.shape[axis]} ' + ('rows.' if axis == 0 else 'columns.'))
            y = self.matrix_prompt('y', validate=validate)
            self.render(solve_system(x, y, orientation), 'solve', equation=orientation)
            return
        model = model or self.choice('Model', ['linear', 'polynomial', 'ridge', 'ridge-polynomial'], 'linear')
        print(f'\nRegression model: {model}')
        print('Target types: numeric = regression; sign = -1/+1 classes; onehot = labels or one-hot rows.')
        target = self.choice('Target type', ['numeric', 'sign', 'onehot'], 'numeric')
        options = dict(model=model, target=target)
        print('Bias adds a column of ones to both training and test data. Choose no if your data already includes it.')
        options['bias'] = self.choice('Add bias (intercept)?', ['yes', 'no'], 'yes') == 'yes'
        if 'polynomial' in model:
            options['degree'] = self.number('Polynomial degree', 2, integer=True)
        if 'ridge' in model:
            print('Lambda controls ridge regularization. Use 0 for ordinary least squares.')
            options['ridge_lambda'] = self.number('Lambda', 1.0)
            options['penalize_bias'] = True
            if options['bias']:
                options['penalize_bias'] = self.choice('Penalize intercept?', ['yes', 'no'], 'yes') == 'yes'
            forms = ['auto', 'primal', 'dual'] if options['penalize_bias'] or not options['bias'] else ['auto', 'primal']
            print('Ridge form: auto = choose by matrix shape; primal = feature-space; dual = sample-space.')
            if 'dual' not in forms:
                print('An unpenalized intercept requires auto or primal.')
            options['form'] = self.choice('Ridge form', forms, 'auto')
        x = self.matrix_prompt('X')
        def validate(y):
            if y.shape[0] != x.shape[0]:
                raise ValueError(f'y needs {x.shape[0]} rows.')
            encode_targets(y, target)
        y = self.matrix_prompt('y', mixed=target == 'onehot', validate=validate)
        x_test = None
        y_test = None
        if self.choice('Predict X_test now?', ['yes', 'no'], 'no') == 'yes':
            print('Enter test features in the same column order as X. The selected bias is added automatically.')
            def validate_test(matrix):
                if matrix.shape[1] != x.shape[1]:
                    raise ValueError(f'X_test needs {x.shape[1]} feature columns, matching X.')
            x_test = self.matrix_prompt('X_test', validate=validate_test)
            if self.choice('Provide y_test to calculate test errors?', ['yes', 'no'], 'no') == 'yes':
                training_targets, labels = encode_targets(y, target)
                def validate_test_targets(matrix):
                    encoded, _ = encode_targets(matrix, target, labels)
                    if encoded.shape != (x_test.shape[0], training_targets.shape[1]):
                        raise ValueError('y_test must match the test sample count and training target columns.')
                y_test = self.matrix_prompt('y_test', mixed=target == 'onehot', validate=validate_test_targets)
        self.fit(x, y, predict=x_test, y_test=y_test, **options)
        print('Use predict followed by data or a matrix name for new predictions.')

    def execute(self, line):
        if '=' in line:
            name, text = line.split('=', 1)
            self.save(name.strip(), parse_matrix(text, mixed=True))
            return
        parts = line.split()
        command, args = parts[0].lower(), parts[1:]
        if command in ('help', 'back'):
            print(HELP)
        elif command in ('1', '2', '3'):
            self.guided(command)
        elif command in ('4', '5', '6', '7', '8', 'pearson', 'gradient', 'impurity', 'tree', 'kmeans') and not args:
            from extra_session import guided, KINDS
            guided(self, KINDS.get(command, command))
        elif command in ('9', 'cache') and not args:
            from extra_session import cache_menu
            cache_menu(self)
        elif command in ('linear', 'polynomial', 'poly', 'ridge', 'ridge-polynomial', 'ridgepoly') and not args:
            self.guided('3', model={'poly': 'polynomial', 'ridgepoly': 'ridge-polynomial'}.get(command, command))
        elif command == 'history':
            for i, item in enumerate(self.history, 1):
                print(f'{i:3}  {item}')
        elif command == 'show' and not args:
            if not self.matrices:
                print('No saved matrices yet. Use set X or X = 1; 2; 3.')
            for name, matrix in self.matrices.items():
                print(f'  {name:16} {matrix.shape[0]} x {matrix.shape[1]}')
        elif command in ('show', 'transpose', 'set', 'edit', 'array') and len(args) == 1:
            name = args[0]
            if command in ('set', 'edit'):
                if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
                    raise ValueError('Use a matrix name such as X or y_test.')
                self.matrix_prompt(name, mixed=True, replace=True)
            elif command == 'array':
                self.render(inspect_array(self.resolve(name)), 'array')
            elif name not in self.matrices:
                raise ValueError(f'No matrix named {name}.')
            elif command == 'transpose':
                self.save(name, self.matrices[name].T)
            else:
                matrix = self.matrices[name]
                print(f'{name} ({matrix.shape[0]} x {matrix.shape[1]})')
                print(np.array2string(matrix, precision=4, threshold=np.inf))
        elif command == 'solve' and len(args) in (2, 3):
            orientation = args[2] if len(args) == 3 else 'right'
            if orientation not in ('right', 'left'):
                raise ValueError('Use solve X y [right|left].')
            self.render(solve_system(self.resolve(args[0]), self.resolve(args[1]), orientation), 'solve', equation=orientation)
        elif command in ('linear', 'ridge', 'poly', 'ridgepoly'):
            counts = dict(linear=2, ridge=3, poly=3, ridgepoly=4)
            if len(args) != counts[command]:
                raise ValueError('Use linear X y; ridge X y lambda; poly X y degree; ridgepoly X y degree lambda.')
            options = dict(model={'poly': 'polynomial', 'ridgepoly': 'ridge-polynomial'}.get(command, command))
            if command in ('poly', 'ridgepoly'):
                options['degree'] = int(args[2])
            if command in ('ridge', 'ridgepoly'):
                options['ridge_lambda'] = float(args[-1])
            self.fit(self.resolve(args[0]), self.resolve(args[1]), **options)
        elif command == 'predict' and args:
            if self.last_fit is None:
                raise ValueError('Fit a regression model first.')
            x, y, options = self.last_fit
            prediction_input = self.resolve(line.split(maxsplit=1)[1])
            result = regress(x, y, predict=prediction_input, **options)
            self.render(result, 'regress', model=options.get('model', 'linear'))
        else:
            raise ValueError('Unknown command or arguments. Type help to see examples.')

    def run(self):
        print(HELP)
        while True:
            try:
                print('\nChoose: 1 array | 2 solve | 3 regression | q/quit/exit')
                from extra_session import MENU
                print(MENU)
                print('Guided commands: pearson | gradient | impurity | tree | kmeans | cache')
                print('Commands: NAME = data | set NAME | edit NAME | show [NAME] | transpose NAME')
                print('          array NAME | solve X y [right|left] | linear X y | ridge X y LAMBDA')
                print('          poly X y DEGREE | ridgepoly X y DEGREE LAMBDA | predict DATA_OR_NAME')
                print('          history | help | back | Enter = stay here')
                print('Full model prompts: linear | polynomial (poly) | ridge | ridge-polynomial (ridgepoly)')
                line = self.reader('ee> ').strip()
                if not line:
                    continue
                if line.lower() in ('q', 'quit', 'exit'):
                    print('Session ended.')
                    return 0
                self.history.append(line)
                self.execute(line)
            except Back:
                print('Returned to main prompt. Saved matrices are available.')
            except (ValueError, TypeError, OSError, np.linalg.LinAlgError) as exc:
                print(f'Error: {exc}')
            except KeyboardInterrupt:
                print('\nCancelled. Type q to quit.')
            except EOFError:
                print('\nSession ended.')
                return 0


def run_session(display):
    return Session(display).run()
