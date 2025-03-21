@echo off
echo ===================================
echo Telegram Trading Bot Runner
echo ===================================
echo.

:start
echo Starting bot at %time% on %date%
echo.
python main.py
echo.
echo Bot stopped at %time%
echo Restarting in 5 seconds...
timeout /t 5 /nobreak > nul
goto start
