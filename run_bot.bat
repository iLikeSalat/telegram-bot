@echo off
echo ===================================
echo Telegram Trading Bot Runner
echo ===================================
echo.

if not exist logs mkdir logs

:start
echo [%date% %time%] Starting bot... >> logs\runner.log
echo Starting bot at %time% on %date%
echo.
python main.py
set EXIT_CODE=%ERRORLEVEL%
echo.
echo Bot stopped at %time% with exit code %EXIT_CODE%
echo [%date% %time%] Bot stopped with exit code %EXIT_CODE% >> logs\runner.log

if %EXIT_CODE% EQU 0 (
    echo Normal shutdown, restarting in 5 seconds...
    echo [%date% %time%] Normal shutdown, restarting... >> logs\runner.log
) else (
    echo Error detected (code %EXIT_CODE%), restarting in 10 seconds...
    echo [%date% %time%] Error detected (code %EXIT_CODE%), restarting... >> logs\runner.log
    timeout /t 10 /nobreak > nul
    goto start
)

timeout /t 5 /nobreak > nul
goto start
