import contextlib
import io
import unittest
import numpy as np
from interactive import Session
from midterm_cli import display


class InteractiveTests(unittest.TestCase):
    def session(self, lines):
        inputs = iter(lines)
        outputs = []
        session = Session(lambda result, args: outputs.append((result, args)), reader=lambda prompt: next(inputs))
        with contextlib.redirect_stdout(io.StringIO()):
            session.run()
        return session, outputs

    def test_reuse_and_prediction_snapshot(self):
        session, outputs = self.session(['X = 0; 1; 2', 'y = 1; 3; 5', 'linear X y',
                                         'y = 100; 100; 100', 'predict 3; 4', 'ridge X y 0.1', 'q'])
        np.testing.assert_allclose(outputs[1][0]['test_predicted'], [[7], [9]])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(session.history), 6)

    def test_multiline_edit_transpose_and_back(self):
        session, outputs = self.session(['set X', '1 2', '3 4', '', 'transpose X',
                                         'edit X', 'back', 'array X', 'q'])
        np.testing.assert_equal(session.numeric(session.matrices['X']), [[1, 3], [2, 4]])
        self.assertEqual(len(outputs), 1)

    def test_guided_retry_keeps_x(self):
        session, outputs = self.session(['2', '', '1 2; 3 4', '5', '5; 11', 'q'])
        np.testing.assert_allclose(outputs[0][0]['w'], [[1], [2]])
        np.testing.assert_equal(session.matrices['X'], [[1, 2], [3, 4]])

    def test_guided_regression_defaults(self):
        session, outputs = self.session(['3', '', '', '', '0; 1; 2', '1; 3; 5', '', 'q'])
        np.testing.assert_allclose(outputs[0][0]['w'], [[1], [2]], atol=1e-12)
        self.assertIsNotNone(session.last_fit)

    def test_guided_bias_prediction_and_test_error(self):
        session, outputs = self.session(['3', 'linear', 'numeric', 'yes', '0; 1; 2', '1; 3; 5',
                                         'yes', '3 4', '3; 4', 'yes', '7; 10', 'predict 5', 'q'])
        np.testing.assert_allclose(outputs[0][0]['test_predicted'], [[7], [9]])
        np.testing.assert_allclose(outputs[0][0]['test_errors']['mse'], [[0.5]])
        np.testing.assert_allclose(outputs[1][0]['test_predicted'], [[11]])
        self.assertEqual(session.matrices['X_test'].shape, (2, 1))

    def test_guided_no_bias_prediction(self):
        _, outputs = self.session(['3', 'linear', 'numeric', 'no', '1; 2', '2; 4',
                                   'yes', '3', 'no', 'q'])
        self.assertEqual(outputs[0][0]['w'].shape, (1, 1))
        np.testing.assert_allclose(outputs[0][0]['test_predicted'], [[6]])

    def test_all_model_prompts_and_prediction(self):
        from calculations import regress, parse_matrix
        for model in ('linear', 'polynomial', 'ridge', 'ridge-polynomial'):
            with self.subTest(model=model):
                settings = ['numeric', 'yes']
                if 'polynomial' in model:
                    settings += ['0', '2']  # Retry invalid degree.
                if 'ridge' in model:
                    settings += ['-1', 'nan', '0.5', 'no', 'dual', 'primal']
                session, outputs = self.session([model, *settings, '0; 1; 2', '1; 4; 9',
                                                 'yes', '3', 'yes', '16', 'q'])
                options = session.last_fit[2]
                self.assertEqual(options['model'], model)
                if 'ridge' in model:
                    self.assertEqual(options['ridge_lambda'], 0.5)
                    self.assertFalse(options['penalize_bias'])
                    self.assertEqual(options['form'], 'primal')
                expected = regress(parse_matrix('0; 1; 2'), parse_matrix('1; 4; 9'),
                                   predict=parse_matrix('3'), **options)
                np.testing.assert_allclose(outputs[0][0]['test_predicted'], expected['test_predicted'])
                self.assertIn('test_errors', outputs[0][0])

    def test_ridge_without_bias_and_zero_lambda(self):
        session, outputs = self.session(['ridge', 'numeric', 'no', '0', 'dual',
                                         '1; 2', '2; 4', 'yes', '3', 'no', 'q'])
        self.assertFalse(session.last_fit[2]['bias'])
        np.testing.assert_allclose(outputs[0][0]['test_predicted'], [[6]])

    def test_errors_do_not_exit_or_destroy_model(self):
        session, outputs = self.session(['predict 3', 'X = 0; 1', 'y = 1; 3', 'linear X y',
                                         'ridge X y -1', 'predict 2', 'q'])
        np.testing.assert_allclose(outputs[-1][0]['test_predicted'], [[5]])
        self.assertEqual(session.last_fit[2]['model'], 'linear')

    def test_full_display_and_eof(self):
        lines = iter(['X = 0; 1', 'y = 1; 3', 'poly X y 2', 'predict 2'])
        def read(prompt):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(Session(display, reader=read).run(), 0)
        for label in ('Design matrix', 'Coefficients', 'Training errors', 'Test design matrix', 'Calculation reference', 'Session ended'):
            self.assertIn(label, output.getvalue())


if __name__ == '__main__':
    unittest.main()
