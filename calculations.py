"""Numerical operations independent of the terminal interface."""
import ast
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def parse_matrix(text, mixed=False):
    text = text.strip().replace("−", "-")
    if not text:
        raise ValueError("Input is empty.")
    if text.startswith("["):
        try:
            values = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Invalid bracketed array. Use [[1, 2], [3, 4]].") from exc
    else:
        values = [row.replace(",", " ").split() for row in text.replace(";", "\n").splitlines() if row.strip()]
    try:
        array = np.asarray(values, dtype=object if mixed else float)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or not all(array.shape):
            raise ValueError()
    except (TypeError, ValueError) as exc:
        raise ValueError("Use a rectangular matrix with valid values and equally sized rows.") from exc
    if not mixed and not np.isfinite(array).all():
        raise ValueError("Values must be finite numbers (no NaN or infinity).")
    return array


def inspect_array(a):
    rows, cols = a.shape
    rank = int(np.linalg.matrix_rank(a))
    result = dict(matrix=a, shape=list(a.shape), rank=rank, transpose=a.T,
                  determinant=float(np.linalg.det(a)) if rows == cols else None,
                  inverse=None, left_inverse=None, right_inverse=None)
    if rows == cols and rank == rows:
        result['inverse'] = np.linalg.solve(a, np.eye(rows))
    if rank == cols:
        result['left_inverse'] = np.linalg.pinv(a)
    if rank == rows:
        result['right_inverse'] = np.linalg.pinv(a)
    return result


def errors(actual, predicted):
    if actual.shape != predicted.shape:
        raise ValueError(f"Target shape {actual.shape} does not match prediction shape {predicted.shape}.")
    residual = predicted - actual
    mse = np.mean(residual ** 2, axis=0, keepdims=True)
    return dict(residual=residual, sse=np.sum(residual ** 2, axis=0, keepdims=True), mse=mse, rmse=np.sqrt(mse))


def solve_system(x, y, equation='right'):
    a, b = (x, y) if equation == 'right' else (x.T, y.T)
    if a.shape[0] != b.shape[0]:
        dimension = 'rows' if equation == 'right' else 'columns'
        raise ValueError(f"X and y must have the same number of {dimension}: got {a.shape[0]} and {b.shape[0]}.")
    w, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    predicted = a @ w
    consistent = bool(np.allclose(predicted, b, rtol=1e-9, atol=1e-10))
    status = ('Unique exact solution' if rank == a.shape[1] else 'Infinitely many exact solutions; showing the minimum-norm solution') if consistent else 'Least-squares solution; showing the minimum-norm minimizer'
    rows, columns = a.shape
    system_type = 'Overdetermined' if rows > columns else 'Underdetermined' if rows < columns else 'Evenly determined (square)'
    solution_type = 'Exact solution' if consistent else 'No exact solution; approximate least-squares solution returned'
    if rank < min(x.shape):
        inverse_type = 'Moore-Penrose pseudoinverse of X (no left or right inverse exists)'
        formula = 'w = pinv(X) @ y' if equation == 'right' else 'w = y @ pinv(X)'
    elif x.shape[0] == x.shape[1]:
        inverse_type = 'Ordinary inverse of X (both left and right inverse)'
        formula = 'w = X^(-1) @ y' if equation == 'right' else 'w = y @ X^(-1)'
    elif x.shape[0] > x.shape[1]:
        inverse_type = 'Left inverse of X (full column rank)'
        formula = 'w = (X.T @ X)^(-1) @ X.T @ y' if equation == 'right' else 'w = y @ (X.T @ X)^(-1) @ X.T'
    else:
        inverse_type = 'Right inverse of X (full row rank)'
        formula = 'w = X.T @ (X @ X.T)^(-1) @ y' if equation == 'right' else 'w = y @ X.T @ (X @ X.T)^(-1)'
    target_statuses = [
        ('Unique exact solution' if rank == columns else 'Infinitely many exact solutions')
        if np.allclose(predicted[:, i], b[:, i], rtol=1e-9, atol=1e-10)
        else 'No exact solution; least-squares approximation'
        for i in range(b.shape[1])
    ]
    result = dict(equation='Xw = y' if equation == 'right' else 'wX = y', x=x, y=y,
                  system_type=system_type, solution_type=solution_type, inverse_type=inverse_type,
                  weight_formula=formula, target_statuses=target_statuses,
                  status=status, rank=int(rank), equations=a.shape[0], unknowns=a.shape[1],
                  w=w, predicted=predicted, **errors(b, predicted))
    if equation == 'left':
        for key in ('w', 'predicted', 'residual', 'sse', 'mse', 'rmse'):
            result[key] = result[key].T
    return result


def encode_targets(y, mode, labels=None):
    if mode != 'onehot':
        try:
            numeric = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric and sign targets must contain numbers.") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("Targets must be finite numbers.")
        if mode == 'sign' and not np.isin(numeric, [-1, 1]).all():
            raise ValueError("Sign classification requires targets of -1 or 1.")
        return numeric, None
    if y.shape[1] > 1:
        numeric = np.asarray(y, dtype=float)
        if not np.isin(numeric, [0, 1]).all() or not np.all(numeric.sum(axis=1) == 1):
            raise ValueError("One-hot rows must contain exactly one 1 and otherwise 0s.")
        if labels is not None and len(labels) != y.shape[1]:
            raise ValueError("One-hot target column count does not match the training classes.")
        return numeric, labels if labels is not None else list(range(y.shape[1]))
    values = [str(v) for v in y[:, 0]]
    labels = list(dict.fromkeys(values)) if labels is None else labels
    mapping = {str(label): i for i, label in enumerate(labels)}
    if any(value not in mapping for value in values):
        raise ValueError("Test targets contain an unknown class label.")
    return np.eye(len(labels))[[mapping[value] for value in values]], labels


def regress(x, y, model='linear', degree=2, bias=True, ridge_lambda=1.0,
            form='auto', penalize_bias=True, target='numeric', predict=None, y_test=None):
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("Lambda must be a finite nonnegative number.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of rows: got {x.shape[0]} and {y.shape[0]}.")
    if y_test is not None and predict is None:
        raise ValueError("Test targets require prediction inputs (--predict or --predict-file).")
    polynomial = model in ('polynomial', 'ridge-polynomial')
    ridge = model in ('ridge', 'ridge-polynomial')
    if polynomial:
        poly = PolynomialFeatures(degree, include_bias=bias)
        design = poly.fit_transform(x)
        names = poly.get_feature_names_out([f'x{i+1}' for i in range(x.shape[1])]).tolist()
        transform = poly.transform
    else:
        transform = lambda a: np.column_stack((np.ones(a.shape[0]), a)) if bias else a
        design = transform(x)
        names = (['1'] if bias else []) + [f'x{i+1}' for i in range(x.shape[1])]
    targets, labels = encode_targets(y, target)
    method = 'Least squares (minimum norm)'
    rank = int(np.linalg.matrix_rank(design))
    if ridge and ridge_lambda > 0:
        resolved = ('dual' if design.shape[0] < design.shape[1] else 'primal') if form == 'auto' else form
        if bias and not penalize_bias:
            if form == 'dual':
                raise ValueError("Use --form primal or auto when excluding the bias from the penalty.")
            resolved = 'primal'
        penalty = np.ones(design.shape[1])
        if bias and not penalize_bias:
            penalty[0] = 0
        if resolved == 'dual':
            w = design.T @ np.linalg.solve(design @ design.T + ridge_lambda * np.eye(design.shape[0]), targets)
            inverse_type = 'Regularized right-inverse form (dual ridge; not an exact inverse)'
            weight_formula = 'w = A.T @ (A @ A.T + lambda * I)^(-1) @ y'
            computation = 'Linear-system solve; no explicit matrix inverse is formed.'
        else:
            # Augmented least squares avoids squaring the design matrix's condition number.
            augmented = np.vstack((design, np.diag(np.sqrt(ridge_lambda * penalty))))
            w = np.linalg.lstsq(augmented, np.vstack((targets, np.zeros((design.shape[1], targets.shape[1])))), rcond=None)[0]
            inverse_type = 'Regularized left-inverse form (primal ridge; not an exact inverse)'
            weight_formula = 'w = (A.T @ A + lambda * D)^(-1) @ A.T @ y'
            computation = 'Augmented least squares; D is identity, with D[0,0]=0 if the intercept is unpenalized.'
        method = f'Ridge {resolved}, lambda={ridge_lambda:g}; bias penalty={bias and penalize_bias}'
    else:
        w = np.linalg.lstsq(design, targets, rcond=None)[0]
        rows, columns = design.shape
        if rank < min(rows, columns):
            inverse_type = 'Moore-Penrose pseudoinverse (rank deficient; no left or right inverse)'
            weight_formula = 'w = pinv(A) @ y'
        elif rows == columns:
            inverse_type = 'Ordinary inverse (both left and right inverse)'
            weight_formula = 'w = A^(-1) @ y'
        elif rows > columns:
            inverse_type = 'Left inverse (full column rank)'
            weight_formula = 'w = (A.T @ A)^(-1) @ A.T @ y'
        else:
            inverse_type = 'Right inverse (full row rank; minimum-norm solution)'
            weight_formula = 'w = A.T @ (A @ A.T)^(-1) @ y'
        computation = 'Computed with least squares for numerical stability; the formula describes the equivalent inverse.'
    fitted = design @ w
    def classes(scores):
        if target == 'sign':
            return np.sign(scores)
        if target == 'onehot':
            return np.asarray(labels, dtype=object)[np.argmax(scores, axis=1)].reshape(-1, 1)
        return None
    result = dict(model=model, samples=x.shape[0], features=x.shape[1], bias=bias,
                  target=target, method=method, design=design, feature_names=names,
                  design_rank=rank, inverse_type=inverse_type, weight_formula=weight_formula, computation=computation,
                  w=w, actual=targets, predicted=fitted, classes=classes(fitted), labels=labels,
                  **errors(targets, fitted))
    if predict is not None:
        if predict.shape[1] != x.shape[1]:
            raise ValueError(f"Prediction inputs need {x.shape[1]} columns; got {predict.shape[1]}.")
        test_design = transform(predict)
        test_pred = test_design @ w
        result.update(test_input=predict, test_design=test_design, test_predicted=test_pred, test_classes=classes(test_pred))
        if y_test is not None:
            test_targets, _ = encode_targets(y_test, target, labels)
            result['test_errors'] = errors(test_targets, test_pred)
    return result
