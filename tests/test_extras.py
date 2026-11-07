import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from calculations import parse_matrix
from extras import calculate, expression
from interactive import Session


class ExtraTests(unittest.TestCase):
    def test_pearson(self):
        r = calculate('pearson', x=parse_matrix('1 2 3'), y=parse_matrix('2 4 6'))
        np.testing.assert_allclose(r['pearson'], [[1]])
        self.assertEqual(r['observations'], 3)
        with self.assertRaises(ValueError):
            calculate('pearson', x=parse_matrix('1; 1'), y=parse_matrix('1; 2'))

    def test_gradient(self):
        r = calculate('gradient', cost='x^2 + y^2', initial=[1, 2], rate=0.1, iterations=2)
        np.testing.assert_allclose(r['final_parameters'], [[0.64], [1.28]])
        self.assertEqual(len(r['history']), 3)
        r = calculate('gradient', cost='sin(x)^2', initial=[3], rate=0.1, iterations=1)
        self.assertAlmostEqual(r['history'][0]['gradient'][0], 2*np.sin(3)*np.cos(3))
        with self.assertRaises(ValueError):
            expression('__import__("os").system("echo bad")')
        with self.assertRaises(ValueError):
            calculate('gradient', cost='log(x)', initial=[-1])

    def test_impurity(self):
        r = calculate('impurity', counts=parse_matrix('3 4; 6 0'), measure='all')
        self.assertAlmostEqual(r['gini']['weighted'], (7/13)*(24/49))
        self.assertAlmostEqual(r['misclassification']['weighted'], 3/13)
        self.assertEqual(r['entropy']['per_node'][1], 0)
        with self.assertRaises(ValueError):
            calculate('impurity', counts=parse_matrix('-1 2'), measure='all')

    def test_tree(self):
        r = calculate('tree', x=parse_matrix('1; 2; 3; 4'), y=parse_matrix('0; 0; 10; 10'), threshold=2, best=True)
        self.assertEqual(r['best_split']['threshold'], 2.5)
        self.assertEqual(r['best_split']['overall_mse'], 0)
        self.assertEqual(len(r['candidates']), 3)
        self.assertEqual(r['root_mse'], 25)

    def test_kmeans(self):
        r = calculate('kmeans', x=parse_matrix('1 1; 1.5 2; 5 7; 6 8'), centroids=parse_matrix('1 1; 5 7'))
        np.testing.assert_allclose(r['final_centroids'], [[1.25, 1.5], [5.5, 7.5]])
        np.testing.assert_equal(r['clusters'], [[1], [1], [2], [2]])
        self.assertEqual(r['stop'], 'Converged')
        self.assertAlmostEqual(r['history'][-1]['cost'], 1.625)

    def test_guided_features(self):
        cases = [
            (['4', '1 2 3', '2 4 6', 'q'], 'pearson_pearson'),
            (['5', 'x^2', '1', '0.1', '2', 'no', 'q'], 'gradient_final_parameters'),
            (['6', 'all', '3 4; 6 0', 'q'], 'impurity_probabilities'),
            (['7', '1; 2; 3', '1; 2; 9', '', 'yes', 'q'], None),
            (['8', '1; 2; 9', '1; 9', '', 'q'], 'kmeans_clusters'),
        ]
        for lines, name in cases:
            with self.subTest(lines=lines):
                inputs = iter(lines)
                s = Session(lambda *args: None, reader=lambda prompt: next(inputs))
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    s.run()
                self.assertNotIn('Error:', output.getvalue())
                if name:
                    self.assertIn(name, s.matrices)

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory)/'session.json')
            inputs = iter(['X = 1 2; 3 4', '9', 'save', filename, '9', 'clear', 'yes', '9', 'load', filename, 'q'])
            s = Session(lambda *args: None, reader=lambda prompt: next(inputs))
            with contextlib.redirect_stdout(io.StringIO()):
                s.run()
            np.testing.assert_equal(s.numeric(s.matrices['X']), [[1, 2], [3, 4]])

    def test_all_new_cli_json(self):
        examples = [
            ['pearson', '--x', '1; 2; 3', '--y', '2; 4; 6'],
            ['gradient', '--cost', 'x^2', '--initial', '1', '--iterations', '2'],
            ['impurity', '--counts', '3 4; 6 0'],
            ['tree', '--x', '1; 2; 3', '--y', '1; 2; 9'],
            ['kmeans', '--x', '1; 2; 9', '--centroids', '1; 9'],
        ]
        script = str(Path(__file__).resolve().parents[1]/'midterm_cli.py')
        for args in examples:
            with self.subTest(command=args[0]):
                result = subprocess.run([sys.executable, script, *args, '--json'], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)
