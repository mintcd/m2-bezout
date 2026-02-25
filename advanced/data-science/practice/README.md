# FLOP count for sklearn models

The experiment counts FLOPs for the sklearn pipelines given in `TP1.ipynb`. The library `pyPAPI` does not work on Windows and WSL.

This solution uses [Intel® Software Development Emulator](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html) for Intel's CPU architectures. In particular, it was ran on


Types of counted operations.

1) Based on precision: single (32 bits) and double (64 bits)
2) Scalar (general registers) and packed (SIMD) 

## Execution

1) Download SDE and extract inside `C:\\` (otherwise modify the path in `profiler.ps1`) 
2) Navigate to this `practice` folder
3) Profile the models
  ```bash
   PowerShell -ExecutionPolicy Bypass -File .\profiler.ps1
  ```

  Notes: it would take much longer (10x) than actual training.
3) Analyze the result.
  ```bash
  python analyze.py
  ```