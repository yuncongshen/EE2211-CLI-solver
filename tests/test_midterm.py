import json
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from calculations import parse_matrix, inspect_array, solve_system, regress


class CalculatorTests(unittest.TestCase):
    def test_solve_system_classification(self):
        cases = [
            ('1; 1', '2; 2', 'Overdetermined', 'Exact solution', 'Left inverse'),
            ('1; 1', '1; 3', 'Overdetermined', 'No exact solution', 'Left inverse'),
            ('1 1', '2', 'Underdetermined', 'Exact solution', 'Right inverse'),
            ('1 0; 0 1', '1; 2', 'Evenly determined', 'Exact solution', 'Ordinary inverse'),
            ('1 1; 2 2', '1; 3', 'Evenly determined', 'No exact solution', 'Moore-Penrose'),
            ('1 1 0; 2 2 0', '1; 3', 'Underdetermined', 'No exact solution', 'Moore-Penrose'),
        ]
        for x, y, system, solution, inverse in cases:
            with self.subTest(x=x, y=y):
                r = solve_system(parse_matrix(x), parse_matrix(y))
                self.assertTrue(r['system_type'].startswith(system))
                self.assertTrue(r['solution_type'].startswith(solution))
                self.assertTrue(r['inverse_type'].startswith(inverse))
        r = solve_system(parse_matrix('1; 1'), parse_matrix('2'), 'left')
        self.assertEqual(r['system_type'], 'Underdetermined')
        self.assertTrue(r['inverse_type'].startswith('Left inverse'))
        np.testing.assert_allclose(r['w'], [[1, 1]])
        r = solve_system(parse_matrix('1 1'), parse_matrix('1 3'), 'left')
        self.assertEqual(r['system_type'], 'Overdetermined')
        self.assertTrue(r['inverse_type'].startswith('Right inverse'))
        self.assertTrue(r['solution_type'].startswith('No exact solution'))
        r = solve_system(parse_matrix('1; 1'), parse_matrix('2 1; 2 3'))
        self.assertTrue(r['target_statuses'][0].startswith('Unique exact'))
        self.assertTrue(r['target_statuses'][1].startswith('No exact'))

    def test_inverse_form_labels(self):
        cases = [
            ('1 0; 0 1; 1 1', 'Left inverse'),
            ('1 0 1; 0 1 1', 'Right inverse'),
            ('1 0; 0 1', 'Ordinary inverse'),
            ('1 1; 2 2; 3 3', 'Moore-Penrose pseudoinverse'),
        ]
        for raw, label in cases:
            x = parse_matrix(raw)
            with self.subTest(label=label):
                result = regress(x, np.ones((x.shape[0], 1)), bias=False)
                self.assertTrue(result['inverse_type'].startswith(label))
        for form, label in [('primal', 'Regularized left'), ('dual', 'Regularized right')]:
            result = regress(np.eye(2), np.ones((2, 1)), model='ridge', bias=False, form=form)
            self.assertTrue(result['inverse_type'].startswith(label))
        result = regress(np.eye(2), np.ones((2, 1)), model='ridge', bias=False, ridge_lambda=0)
        self.assertTrue(result['inverse_type'].startswith('Ordinary inverse'))

    def test_parsing(self):
        for raw in ('1 2; 3 4', '1,2\n3,4', '[[1,2],[3,4]]'):
            np.testing.assert_equal(parse_matrix(raw), [[1, 2], [3, 4]])
        for raw in ('1 2; 3', 'nan', '', '[[[1]]]'):
            with self.assertRaises(ValueError):
                parse_matrix(raw)

    def test_inverse(self):
        a = parse_matrix('1 2; 3 4')
        np.testing.assert_allclose(a @ inspect_array(a)['inverse'], np.eye(2), atol=1e-12)
        tall = parse_matrix('1 0; 0 1; 1 1')
        np.testing.assert_allclose(inspect_array(tall)['left_inverse'] @ tall, np.eye(2), atol=1e-12)
        self.assertIsNone(inspect_array(parse_matrix('1 2; 2 4'))['inverse'])

    def test_solve_orientations_and_status(self):
        x = parse_matrix('1 2; 3 4')
        np.testing.assert_allclose(solve_system(x, parse_matrix('5; 11'))['w'], [[1], [2]])
        np.testing.assert_allclose(solve_system(x, parse_matrix('7 10'), 'left')['w'], [[1, 2]])
        r = solve_system(parse_matrix('1 1'), parse_matrix('2'))
        self.assertIn('Infinitely', r['status'])
        np.testing.assert_allclose(r['w'], [[1], [1]])
        r = solve_system(parse_matrix('1; 1'), parse_matrix('1; 3'))
        self.assertIn('Least-squares', r['status'])
        np.testing.assert_allclose(r['mse'], [[1]])

    def test_linear_polynomial_and_test_error(self):
        x = parse_matrix('0; 1; 2; 3')
        r = regress(x, parse_matrix('1; 3; 5; 7'), predict=parse_matrix('4; 5'), y_test=parse_matrix('9; 12'))
        np.testing.assert_allclose(r['w'], [[1], [2]], atol=1e-12)
        np.testing.assert_allclose(r['test_errors']['mse'], [[0.5]])
        r = regress(parse_matrix('0; 1; 2'), parse_matrix('1; 4; 9'), model='polynomial', predict=parse_matrix('3'))
        np.testing.assert_allclose(r['test_predicted'], [[16]])

    def test_ridge_square_primal_dual_and_bias(self):
        x = np.eye(2)
        y = np.array([[2.], [4.]])
        for form in ('auto', 'primal', 'dual'):
            r = regress(x, y, model='ridge', bias=False, ridge_lambda=1, form=form)
            np.testing.assert_allclose(r['w'], [[1], [2]])
        r = regress(np.zeros((3, 1)), np.full((3, 1), 5.), model='ridge', penalize_bias=False)
        np.testing.assert_allclose(r['w'], [[5], [0]], atol=1e-12)

    def test_ridge_polynomial_and_multiple_outputs(self):
        x = parse_matrix('0; 1; 2')
        y = parse_matrix('1 2; 4 8; 9 18')
        r = regress(x, y, model='ridge-polynomial', ridge_lambda=0.5)
        p = r['design']
        expected = np.linalg.solve(p.T @ p + 0.5 * np.eye(p.shape[1]), p.T @ y)
        np.testing.assert_allclose(r['w'], expected)
        r = regress(parse_matrix('1 1; 2 2'), parse_matrix('2; 4'), bias=False)
        np.testing.assert_allclose(r['w'], [[1], [1]])

    def test_classification(self):
        x = parse_matrix('0; 1; 2; 3')
        r = regress(x, parse_matrix('low; low; high; high', mixed=True), target='onehot', predict=parse_matrix('0; 3'))
        self.assertEqual(r['test_classes'].ravel().tolist(), ['low', 'high'])
        r = regress(x, parse_matrix('-1; -1; 1; 1'), target='sign', predict=parse_matrix('0; 3'))
        np.testing.assert_equal(r['test_classes'], [[-1], [1]])

    def test_validation(self):
        with self.assertRaises(ValueError):
            regress(np.ones((3, 1)), np.ones((2, 1)))
        with self.assertRaises(ValueError):
            regress(np.eye(2), np.ones((2, 1)), ridge_lambda=-1)
        with self.assertRaises(ValueError):
            regress(np.eye(2), np.ones((2, 1)), predict=np.ones((1, 3)))

    def test_cli(self):
        script = str(Path(__file__).resolve().parents[1] / 'midterm_cli.py')
        done = subprocess.run([sys.executable, script, 'solve', '--x', '1 2; 3 4', '--y', '5; 11', '--json'], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        np.testing.assert_allclose(json.loads(done.stdout)['w'], [[1], [2]])
        bad = subprocess.run([sys.executable, script, 'solve', '--x', '1 2', '--y', '1; 2'], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn('Error:', bad.stderr)
        self.assertNotIn('Traceback', bad.stderr)


if __name__ == '__main__':
    unittest.main()
