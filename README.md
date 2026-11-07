# EE2211 Calculator — User Guide

A terminal calculator for matrices, linear equations, regression, classification,
correlation, gradient descent, decision trees, and K-means. Calculations run locally.
All results and intermediate tables appear automatically; there is no details mode.

## Contents

1. [Start the app](#start-the-app)
2. [First calculation](#first-calculation)
3. [Input formats and navigation](#input-formats-and-navigation)
4. [Menu 1: Array inspection](#menu-1-array-inspection)
5. [Menu 2: Solve equations](#menu-2-solve-equations)
6. [Menu 3: Regression and classification](#menu-3-regression-and-classification)
7. [Menu 4: Pearson correlation](#menu-4-pearson-correlation)
8. [Menu 5: Gradient descent](#menu-5-gradient-descent)
9. [Menu 6: Classification-tree impurity](#menu-6-classification-tree-impurity)
10. [Menu 7: Regression-tree splits](#menu-7-regression-tree-splits)
11. [Menu 8: K-means](#menu-8-k-means)
12. [Menu 9: Cache and session files](#menu-9-cache-and-session-files)
13. [Interactive command reference](#interactive-command-reference)
14. [Standalone commands and files](#standalone-commands-and-files)
15. [Troubleshooting](#troubleshooting)
16. [Installation, verification, and attribution](#installation-verification-and-attribution)

## Start the app

**Easiest:** double-click `start.cmd` in this folder. It uses the local `.venv`
environment and opens the menu. The environment must already be installed.

Alternatively, open PowerShell and enter:

```powershell
cd C:\Users\sheny\EE2211\midterm
.\.venv\Scripts\ee.exe
```

Or, if you have `uv`:

```powershell
cd C:\Users\sheny\EE2211\midterm
uv run ee
```

The menu offers:

| Number | Feature |
| --- | --- |
| 1 | Array inspection |
| 2 | Solve Xw = y or wX = y |
| 3 | Regression and classification |
| 4 | Pearson correlation |
| 5 | Gradient descent |
| 6 | Classification-tree impurity |
| 7 | Regression-tree splits |
| 8 | K-means |
| 9 | Cache and save/load matrices |
| q | Quit |

**Two places to enter commands:** PowerShell accepts commands beginning with
`.\.venv\Scripts\ee.exe`. Inside the app, the `ee>` prompt accepts menu numbers
and short commands such as `linear X y`. Do not include `ee>` when typing.

## First calculation

At the app's `ee>` prompt, enter these lines individually:

```text
X = 0; 1; 2; 3
y = 1; 3; 5; 7
linear X y
predict 4; 5
```

This saves two column vectors, fits `y = 1 + 2*x1`, and predicts `9` and `11`.
The app shows the design matrix, coefficients, inverse form, training errors,
test design matrix, and predictions. The training MSE is zero to display precision.

To use guided prompts instead, type `3`, choose `linear`, choose `numeric`,
choose `yes` for bias, and enter the same X and y. Choose `yes` for prediction,
enter `4; 5` for X_test, and choose `no` when asked for y_test.

## Input formats and navigation

### Matrices and vectors

| Input | Meaning | Shape |
| --- | --- | --- |
| `1 2; 3 4` | Two rows, two columns | 2 x 2 |
| `1,2; 3,4` | Same matrix with commas | 2 x 2 |
| `1 2 3` | Row vector | 1 x 3 |
| `1; 2; 3` | Column vector | 3 x 1 |
| `[1, 2, 3]` | Flat bracket list becomes a column | 3 x 1 |
| `[[1, 2], [3, 4]]` | Explicit nested rows | 2 x 2 |
| `2.5` | Single value | 1 x 1 |

Use decimal numbers, negatives, or scientific notation such as `1e-3`. Numeric
calculations require finite values: no NaN or infinity. Every row must have the
same number of columns. Matrix entries are numeric literals, not expressions:
enter `0.5`, not `1/2`.

For regression and K-means, **rows are samples and columns are features**.
A dataset of four samples with one feature should be `0; 1; 2; 3`, not `0 1 2 3`.

At interactive prompts, type the data directly without enclosing the whole input
in quotes. In PowerShell commands, quote inline matrices so spaces and semicolons
stay together as a single argument.

### Paste multiple rows

At `ee>`, type:

```text
set X
```

Then paste or type:

```text
1 2
3 4
```

Press Enter on an empty line to finish. `edit X` works the same way and replaces
all of X. A matrix prompt also opens multiline entry when you press Enter and
there is no saved matrix with that prompt's name.

### Defaults and saved values

- Choice prompts show every supported option. Type the option or press Enter for the displayed default.
- At a matrix prompt, Enter **reuses the named matrix if it exists**. Check the displayed dimensions before accepting it.
- Type another saved matrix name to reuse its data at a prompt.
- `show` lists saved names and shapes; `show X` displays X.
- `transpose X` replaces X with its transpose.
- Names are case-sensitive: `X` and `x` are different names. Names begin with a letter and contain only letters, digits, and underscores.
- Invalid guided input is normally retried at the same prompt. A calculation-level error returns to the main prompt; saved inputs remain available.
- `back` cancels the current guided operation. At the main prompt it shows help.
- Ctrl+C cancels the current operation. Use `q`, `quit`, or `exit` at the main prompt to quit.
- `history` lists commands entered at the main prompt, including failed attempts; it does not record every guided answer or replay commands.

## Menu 1: Array inspection

Choose `1`, enter A, and read its shape, rank, determinant, transpose, and available
inverses. Alternatively save a matrix and type `array X`.

Example input: `1 2; 3 4`.

Expected highlights:

```text
Shape: 2 x 2
Rank: 2
Determinant: -2
Inverse:
  -2.0   1.0
   1.5  -0.5
```

A determinant is only shown for a square matrix. A left inverse exists when the
columns are independent; a right inverse exists when the rows are independent.
For an invertible square matrix these are the same ordinary inverse. Unavailable
inverses are labelled rather than forcing an invalid calculation.

## Menu 2: Solve equations

Choose `2`, then select the equation orientation:

| Choice | Equation | Required dimensions |
| --- | --- | --- |
| `right` (default) | Xw = y | X and y have the same number of rows |
| `left` | wX = y | X and y have the same number of columns |

Enter X and y. The app reports:

- Equation and unknown counts, rank, and whether the system is overdetermined, evenly determined, or underdetermined.
- Whether there is a unique exact solution, infinitely many exact solutions, or no exact solution.
- The minimum-norm solution or least-squares approximation returned.
- Left/right/ordinary inverse or pseudoinverse form and the corresponding formula.
- Weights w, predicted y, residuals, SSE, MSE, and RMSE.

For `Xw = y`, try X=`1 2; 3 4` and y=`5; 11`. The unique exact answer is
w=`1; 2`. For `wX = y`, use the same X and y=`7 10`; the answer is w=`1 2`.

To see an approximation, use X=`1; 1`, y=`1; 3`, and `right`. There is no exact
solution; the returned weight is `2`, with predicted y=`2; 2` and MSE=`1`.

System shape alone does not establish whether a solution exists. An overdetermined
system can be exactly consistent; a singular square system can have infinitely
many solutions or no exact solution. Exactness is assessed with floating-point
tolerances. Multiple target columns (or rows for left-oriented solves) have
individual solution-status labels.

The inverse label refers to the original X. With `wX = y`, equation counts use
the transposed system, so a tall X can describe an underdetermined problem while
still having a left inverse. The displayed formula makes the multiplication order
explicit. Computation uses stable least-squares routines rather than forming the
listed inverse directly.

## Menu 3: Regression and classification

Choose `3` to select a model, or type just `linear`, `polynomial`, `ridge`, or
`ridge-polynomial` at `ee>` to open that model's full guided workflow.

### Model settings

| Model | Additional settings |
| --- | --- |
| `linear` | Bias/intercept |
| `polynomial` | Bias and polynomial degree |
| `ridge` | Bias, lambda, intercept penalty when bias is enabled, ridge form |
| `ridge-polynomial` | All of the above |

All models also ask for target type, X, y, optional X_test, and optional y_test.

| Setting | Choices/default | Meaning |
| --- | --- | --- |
| Target type | `numeric` (default), `sign`, `onehot` | Regression, binary sign classification, or one-hot classification |
| Add bias | `yes` (default), `no` | Adds the constant term automatically |
| Polynomial degree | Positive integer; default 2 | Includes powers and interactions up to this degree |
| Lambda | Finite number >= 0; default 1 | Ridge penalty strength; 0 gives ordinary least squares |
| Penalize intercept | `yes` (default), `no` | Whether the bias weight participates in the ridge penalty |
| Ridge form | `auto` (default), `primal`, `dual` | Equivalent regularized formulations where applicable |
| Predict X_test | `yes`, `no` (default) | Predict for new samples |
| Provide y_test | `yes`, `no` (default) | Calculate errors against known test targets |

`auto` chooses dual when the design matrix has fewer samples than parameters,
and primal otherwise. An unpenalized intercept requires primal; the guided menu
only offers `auto` and `primal` in that case. With lambda=0 the least-squares path
is used regardless of the requested ridge form.

**Bias:** choose `yes` when X contains only the measured features. Choose `no` if
X already includes a ones column or the desired model has no intercept. For
polynomial models, supply raw features and let the app generate powers and
interactions; do not manually add a constant column unless you deliberately want
it treated as an input feature.

### Training and prediction walkthrough

For a simple linear model, answer:

| Prompt | Input |
| --- | --- |
| Model | `linear` |
| Target | `numeric` |
| Add bias | `yes` |
| X | `0; 1; 2; 3` |
| y | `1; 3; 5; 7` |
| Predict X_test | `yes` |
| X_test | `4; 5` |
| Provide y_test | `yes` |
| y_test | `9; 12` |

The fitted equation is `y = 1 + 2*x1`, predictions are `9; 11`, and test MSE is
`0.5`. X_test must have the same feature columns, in the same order, as X. The
chosen bias and polynomial transformations are applied automatically to both.

For polynomial regression, try degree 2 with X=`0; 1; 2`, y=`1; 4; 9`, and
X_test=`3; 4`. The model is `1 + 2*x1 + x1^2`, predicting `16; 25`.

For ridge, use the linear example with lambda `0.1`. The penalty changes the
weights; the output identifies the regularized inverse form actually selected.
The default penalizes the intercept as well as other weights.

### Classification

- `sign`: supply targets -1 or +1, such as `-1; -1; 1; 1`. Predicted scores are converted using their sign. An exactly zero score gives class 0.
- `onehot`: supply a label column such as `low; low; high; high`, or one-hot rows such as `1 0; 1 0; 0 1; 0 1`.
- Label classes use first-appearance order. One-hot columns without label names are reported as classes 0, 1, and so on. The largest score determines the predicted class; ties select the first class.
- Test labels must belong to the training classes. Every one-hot row must contain one 1 and otherwise 0s.
- For names containing spaces, use an explicit nested list, for example `[['very low'], ['very high']]`.

Classification is explicitly selected; numeric multi-column targets remain
multi-output regression unless you select `onehot`. Classification SSE/MSE measure
errors in encoded scores, not classification accuracy.

### Reading the report

The design matrix A is X after adding the selected bias and polynomial terms.
Its column labels tell you the order of the coefficients. Results include:

- A and its rank, coefficient table, and a fitted equation for single-output models.
- Inverse form and equivalent weight formula. Full column rank uses a left-inverse form; full row rank uses a right-inverse form; square invertible designs use the ordinary inverse; rank-deficient designs use a pseudoinverse.
- Training actual values, predictions, residuals, and errors per output.
- Transformed test inputs, predictions, optional classes, and optional test errors.

Ridge forms are regularized formulas, not exact left/right inverses. The actual
implementation uses numerical solvers for stability.

After any successful fit, `predict 4; 5` or `predict X_test` predicts with a
snapshot of that fit. Editing X or y afterward does not change the stored model;
fit again to use the edited dataset. This prediction command shows the full
report again. It does not prompt for test targets; use the guided workflow if you
also want to supply y_test.

## Menu 4: Pearson correlation

Choose `4` or type `pearson`. Enter X and y with observations in rows and variables
in columns. For X=`1; 2; 3`, y=`2; 4; 6`, correlation is `1`.

The app shows oriented inputs, observation count, means, population standard
deviations and variances, covariance, and correlations for every X-column/y-column
pair. Correlation-table rows correspond to X variables; columns correspond to y
variables. Covariance and variance use division by N, not N-1.

Matching single-row vectors are transposed automatically. If row counts differ
but column counts match, both inputs are transposed. Check the orientation label
when supplying multiple variables. At least two observations are required, and
constant columns cause an undefined-correlation error.

## Menu 5: Gradient descent

Choose `5` or type `gradient`. Enter:

1. A symbolic cost function, for example `x^2 + y^2`.
2. Initial values in the variable order printed by the app, for example `1 2` for x, y.
3. Learning rate, default `0.1`.
4. Positive iteration count, default `10`.
5. Whether to use a stopping tolerance; if yes, enter a nonnegative tolerance (default `0.000001`).

Supported syntax: numbers, variable names, `+`, `-`, `*`, `/`, `^` or `**`,
parentheses, `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `Abs`, and constants `pi`
and `E`. Use explicit multiplication (`2*x`, not `2x`). `log` is natural logarithm;
trigonometric arguments are radians. `E` is a constant, not a fitted variable.
Variables are sorted by name; always follow the displayed order.

The update is `values_next = values - learning_rate * gradient`. The report shows
the symbolic derivatives and a table containing initial state (step 0), every
update, parameter values, gradients at those values, and cost.

Example: cost `x^2 + y^2`, initial `1 2`, rate `0.1`, two iterations, no tolerance.
Final parameters are `0.64; 1.28` and final cost is `2.048`.

Tolerance stops when **either** maximum absolute parameter change **or** absolute
cost change is at most the selected tolerance. Reaching tolerance is not a proof
of a global minimum. Non-real or non-finite evaluations produce an error; inspect
the function domain or reduce the learning rate if this occurs.

## Menu 6: Classification-tree impurity

Choose `6` or type `impurity`. Select `all`, `gini`, `entropy`, or
`misclassification`, then enter class counts. **Each row is a node; each column
is a class.** Keep class columns in the same order for every node.

Example: `3 4; 6 0` describes two nodes, with 7 and 6 samples respectively.

For class probabilities p:

| Measure | Formula |
| --- | --- |
| Gini | `1 - sum(p^2)` |
| Entropy | `-sum(p * log2(p))`, omitting zero probabilities |
| Misclassification | `1 - max(p)` |

The report shows probabilities, node sizes, impurity per node, and weighted
impurity using node size / total sample count. In the example, node Gini values
are approximately `0.4898` and `0`, with weighted Gini approximately `0.2637`.
Counts must be nonnegative and every node must have a positive total. This tool
evaluates supplied nodes; it does not build a full classification tree.

## Menu 7: Regression-tree splits

Choose `7` or type `tree`. Enter matching x and y vectors with at least two
samples. You may enter a threshold or press Enter to skip it. Then choose whether
to search all candidate splits (default `yes`).

A threshold t puts `x <= t` on the left and `x > t` on the right. Both children
must contain at least one sample. Candidate thresholds are midpoints between
consecutive distinct sorted x values.

Example: x=`1; 2; 3; 4`, y=`0; 0; 10; 10`, threshold=`2.5`.
The root mean is `5`, root MSE is `25`, and both child MSEs are `0`. Best-split
search also selects `2.5`, with weighted MSE `0`.

The report includes root statistics, the chosen split when supplied, child sizes,
child means, child MSEs, weighted MSE, and all candidate thresholds. Tied best
scores select the first candidate in ascending threshold order. If all x values
are equal, the guided workflow shows root statistics without searching. This is
a single-split calculator, not a recursive tree trainer.

## Menu 8: K-means

Choose `8` or type `kmeans`. Enter point rows, initial centroid rows, and an
optional maximum iteration count. The number of initial centroid rows is K.
Press Enter at the iteration prompt to run until convergence.

Example:

```text
Points:             1 1; 1.5 2; 5 7; 6 8
Initial centroids:  1 1; 5 7
Maximum iterations: leave blank
```

Final centroids are `1.25 1.5; 5.5 7.5`, assignments are `1; 1; 2; 2`, and SSE
cost is `1.625`.

The report shows the initial state and each changed set of centroids, point
assignments, distances to each centroid, cost, and stop reason. Points and
centroids must have the same feature count. Distances are Euclidean and cost is
the **sum of squared distances** to assigned centroids, not their mean.

Clusters are numbered from 1. Distance ties choose the first centroid. Empty
clusters retain their previous centroid. Convergence checks whether centroid
coordinates are unchanged within numerical tolerance. The reference convention
marks the preceding history entry converged rather than appending a duplicate
unchanged iteration. If the iteration cap is reached, the output may say
`Iteration limit reached` even when a subsequent check would confirm convergence.
No random initialization is performed: you supply the starting centroids.

## Menu 9: Cache and session files

Choose `9` or type `cache`:

| Action | Behaviour |
| --- | --- |
| `list` | Show saved matrix names and dimensions |
| `show` | Display a selected matrix |
| `delete` | Remove one named matrix |
| `clear` | Clear all matrices after confirmation |
| `save` | Save matrices to a JSON file; confirm if replacing a file |
| `load` | Load a saved JSON file; confirm replacement of matching names |

For example, choose `save` and enter `session.json`. Later, start the app, choose
`load`, and enter the same path. Relative paths are resolved from the process's
working directory; `start.cmd` sets it to this folder. Parent directories must
already exist. Enter paths without wrapping them in quotes at the interactive
prompt, even if they contain spaces.

Inputs and top-level matrix outputs are automatically cached. Common names:

| Name | Contents |
| --- | --- |
| `array_transpose` | Transpose from array inspection |
| `solve_w` | Most recent equation solution |
| `regress_w` | Most recent regression coefficients |
| `regress_design` | Training design matrix |
| `regress_test_predicted` | Most recent regression test predictions |
| `pearson_pearson` | Correlation matrix |
| `gradient_final_parameters` | Final gradient-descent parameters |
| `impurity_probabilities` | Class probabilities per node |
| `kmeans_final_centroids` | Final cluster centroids |
| `kmeans_clusters` | One-based cluster assignments |

Repeated calculations replace matching output names. Outputs that are not
produced by a later calculation are not automatically removed: for example, a
previous `regress_test_predicted` can remain after a fit without test inputs.
Nested histories and scalar summaries are displayed but are not all cached.
Use standalone `--json` output to save a complete calculation report.

Saved cache files contain **matrices only**, not fitted models, settings, or
command history. Load merges into the current cache. To predict in a new session,
load X/y and fit again. Deleting or clearing matrices does not erase the existing
last-fit snapshot within the current session. Quitting discards that snapshot
and all unsaved matrices. Matrix files are not automatically saved on exit.

## Interactive command reference

These commands are entered inside the app, at `ee>`:

| Command | Purpose |
| --- | --- |
| `X = 1 2; 3 4` | Save inline data |
| `set X` / `edit X` | Paste replacement data |
| `show` / `show X` | List matrices / display X |
| `transpose X` | Transpose X in place |
| `array X` | Inspect X |
| `solve X y` | Solve Xw=y |
| `solve X y left` | Solve wX=y |
| `linear`, `polynomial`/`poly`, `ridge`, `ridge-polynomial`/`ridgepoly` | Open full model prompts |
| `linear X y` | Immediate numeric linear fit with bias |
| `poly X y 2` | Immediate numeric degree-2 polynomial fit with bias |
| `ridge X y 0.1` | Immediate ridge fit with bias, lambda=0.1 |
| `ridgepoly X y 2 0.1` | Immediate ridge polynomial fit; degree then lambda |
| `predict 4; 5` / `predict X_test` | Predict with last fitted model |
| `pearson`, `gradient`, `impurity`, `tree`, `kmeans`, `cache` | Open feature prompts |
| `history` | Show main-prompt command history |
| `help` / `back` | Display help at the main prompt |
| `q` / `quit` / `exit` | End session |

Complete short regression commands use numeric targets, bias enabled, and default
ridge settings (including intercept penalty and automatic form selection). Use
the full guided prompts for classification, no bias, or other settings. Short
matrix-operation commands are easiest with saved matrix names; use matrix prompts
or standalone quoted arguments for complex inline matrices.

## Standalone commands and files

These commands run in PowerShell, not at `ee>`. Each launches a separate
calculation without an interactive cache. From this folder:

```powershell
.\.venv\Scripts\ee.exe array "1 2; 3 4"
.\.venv\Scripts\ee.exe solve --x "1 2; 3 4" --y "5; 11"
.\.venv\Scripts\ee.exe solve --equation left --x "1 2; 3 4" --y "7 10"
.\.venv\Scripts\ee.exe regress linear --x "0; 1; 2; 3" --y "1; 3; 5; 7" --predict "4; 5"
.\.venv\Scripts\ee.exe regress polynomial --degree 2 --x "0; 1; 2" --y "1; 4; 9" --predict "3; 4"
.\.venv\Scripts\ee.exe regress ridge --lambda 0.1 --form primal --no-penalize-bias --x "0; 1; 2" --y "1; 3; 5"
.\.venv\Scripts\ee.exe regress ridge-polynomial --degree 2 --lambda 0.1 --x "0; 1; 2" --y "1; 4; 9"
.\.venv\Scripts\ee.exe regress linear --target onehot --x "0; 1; 2; 3" --y "low; low; high; high" --predict "0; 3"
.\.venv\Scripts\ee.exe regress linear --target sign --x "0; 1; 2; 3" --y="-1; -1; 1; 1"
.\.venv\Scripts\ee.exe pearson --x "1; 2; 3" --y "2; 4; 6"
.\.venv\Scripts\ee.exe gradient --cost "x^2 + y^2" --initial "1 2" --rate 0.1 --iterations 2
.\.venv\Scripts\ee.exe impurity --counts "3 4; 6 0" --measure all
.\.venv\Scripts\ee.exe tree --x "1; 2; 3; 4" --y "0; 0; 10; 10" --threshold 2.5
.\.venv\Scripts\ee.exe kmeans --x "1 1; 1.5 2; 5 7; 6 8" --centroids "1 1; 5 7"
```

### Options reference

| Command | Main options |
| --- | --- |
| `array` | Positional matrix or `--file` |
| `solve` | `--x`, `--y`, `--equation right/left` |
| `regress MODEL` | `--x`, `--y`, `--bias`/`--no-bias`, `--degree`, `--lambda`, `--form`, `--penalize-bias`/`--no-penalize-bias`, `--target`, `--predict`, `--y-test` |
| `pearson` | `--x`, `--y` |
| `gradient` | `--cost`, `--initial`, `--rate`, `--iterations`, optional `--tolerance` |
| `impurity` | `--counts`, `--measure all/gini/entropy/misclassification` |
| `tree` | `--x`, `--y`, optional `--threshold`, `--best`/`--no-best` (best enabled by default) |
| `kmeans` | `--x`, `--centroids`, optional `--iterations` |

Every calculation command supports `--help`, `--precision N` (0–15, default 4),
and `--json`. Display precision does not alter calculations. Interactive reports
use four decimal places. Small floating-point residuals may display as zero.

```powershell
.\.venv\Scripts\ee.exe regress --help
.\.venv\Scripts\ee.exe solve --x "1 2; 3 4" --y "5; 11" --precision 8
.\.venv\Scripts\ee.exe solve --x "1 2; 3 4" --y "5; 11" --json > result.json
```

The JSON calculation report contains full-precision results and intermediate
arrays. It is **not** the same format as a cache session file; do not load it via
menu 9. Success exits with code 0; input/calculation errors normally exit with
code 2 and an explanation.

### CSV and text input

Files must be headerless UTF-8 CSV or whitespace-separated text, with one row per
line. For example, `X.csv`:

```text
0
1
2
3
```

And `y.csv`:

```text
1
3
5
7
```

Run:

```powershell
.\.venv\Scripts\ee.exe regress linear --x-file X.csv --y-file y.csv
```

Matrix options have corresponding file options: `--x-file`, `--y-file`,
`--predict-file`, `--y-test-file`, `--initial-file`, `--counts-file`, and
`--centroids-file`. Array inspection uses `--file`. Choose either inline or file
input for each field, not both. These file options apply to standalone commands;
interactive matrix prompts accept typed/pasted data or cached names.

For negative-leading values in named PowerShell arguments, use an equals sign,
for example `--y="-1; 1"`. For an array positional argument, use:

```powershell
.\.venv\Scripts\ee.exe array -- "-1 2; 3 4"
```

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `python` or `ee` is not recognized | Use `.\.venv\Scripts\ee.exe`, `start.cmd`, or `uv run ee` from this folder. |
| `.venv` or its executable is missing | Follow the installation steps below. |
| PowerShell blocks environment activation | Activation is unnecessary; use the full executable path. |
| X and y row counts differ | Use one sample per row; `1; 2; 3` is a column. Check `show X` and `show y`. |
| X_test has the wrong feature count | Use the same raw feature columns as X. Do not add a bias column when the app is adding it. |
| Old inputs are reused | Enter at a matrix prompt reuses its saved value. Type new data or use `edit NAME`. |
| Predictions ignore edited y | Predictions use the last-fit snapshot. Fit the model again after editing data. |
| A matrix name cannot be found | Names are case-sensitive. Use `show` and check X versus x. |
| Pearson correlation is undefined | Remove constant variables and provide at least two observations. |
| Tree threshold leaves an empty side | Choose a threshold within the data range that separates at least two distinct x values. |
| Gradient descent gives non-real/non-finite values | Check the function domain, initial values, and learning rate. |
| K-means reports an iteration limit | Increase the limit or leave it blank in guided mode. |
| Output contains very small errors | Floating-point calculations are approximate; use more display precision if needed. |
| Cache save/load fails | Check the path, existing parent directory, file permissions, and that the file was saved by menu 9. |
| Numeric input `1/2` fails | Use `0.5`. Expressions are supported in gradient cost functions, not ordinary matrix cells. |

Error definitions used throughout the app:

```text
residual = predicted - actual
SSE = sum(residual^2)
MSE = SSE / number of samples
RMSE = sqrt(MSE)
```

Errors are reported per output for multiple targets. For `wX = y`, errors are
reported per target row. An exact fit or zero training error does not guarantee
good predictions on new data.

## Installation, verification, and attribution

The app requires Python 3.11+, NumPy, scikit-learn, and SymPy. A local environment
has been set up in `.venv`. To install/recreate dependencies with uv:

```powershell
cd C:\Users\sheny\EE2211\midterm
uv sync
uv run ee
```

Alternatively, with an available Python executable:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ee.exe
```

Replace `python` with your installed executable if needed (for example `py` or
`python3.14`). Installing dependencies may require internet access; calculations
after installation run locally. Optional environment activation allows the
shorter `ee` command:

```powershell
.\.venv\Scripts\Activate.ps1
ee
```

Run the test suite from this folder:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The original matrix/regression features were independently implemented using
`ee2211-solver` as a feature reference. Additional calculations are adapted from
that MIT-licensed project. Its copyright notice and license are preserved in
[reference_math/LICENSE](reference_math/LICENSE). The app bundles its required
reference calculations locally and does not import from or modify the original
`ee2211-solver` folder. No browser or Streamlit is required.
