@echo off
echo ---------------------------------------------
echo MLbotnew - 2h Coinglass Backtest Execution
echo ---------------------------------------------

REM Create necessary directories
if not exist logs mkdir logs
if not exist data\raw\2h mkdir data\raw\2h
if not exist data\features\2h mkdir data\features\2h
if not exist reports\2h mkdir reports\2h

echo.
echo 1. Downloading 2h Coinglass data (360 days)...
python scripts/run_real_backtest.py --interval 2h --limit 4320

echo.
echo 2. Running backtest with 2h data...
python scripts/run_backtest.py --interval 2h

echo.
echo 3. Displaying backtest results...
python scripts/display_backtest_results.py --interval 2h

echo.
echo Backtest completed! Reports saved to reports/2h/
echo Please check the reports folder for HTML visualization and statistics.
echo.

pause
