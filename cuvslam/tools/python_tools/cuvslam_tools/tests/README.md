# cuvslam_tools Unit Tests

From the repository `src` directory, run all tests in this folder with the Python
environment where `cuvslam_tools` test dependencies are installed:

```bash
PYTHONPATH=tools/python_tools python -m unittest discover \
  -s tools/python_tools/cuvslam_tools/tests \
  -p "test_*.py" \
  -b
```

From `src/tools/python_tools`, run:

```bash
PYTHONPATH=. python -m unittest discover \
  -s cuvslam_tools/tests \
  -p "test_*.py" \
  -b
```

If you use a virtual environment, activate it first or replace `python` with the
path to that environment's Python executable.
