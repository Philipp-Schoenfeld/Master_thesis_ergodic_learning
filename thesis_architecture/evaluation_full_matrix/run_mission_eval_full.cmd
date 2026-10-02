@echo off
rem Self-test first, then (only if it passed) the full replanning-mission evaluation.
rem Resumable: running this file again continues with the rounds that are not stored yet.
rem Results: G:\mission_eval\mission_eval_20261002  (C: has too little free space)
rem Log:     G:\mission_eval\mission_eval_20261002.log
cd /d "%~dp0"
if not exist G:\mission_eval mkdir G:\mission_eval
set LOG=G:\mission_eval\mission_eval_20261002.log
echo [chain] %DATE% %TIME% starting self-test >> %LOG%
python -u test_mission_eval.py >> %LOG% 2>&1
if errorlevel 1 (
    echo [chain] %DATE% %TIME% self-test FAILED - full run not started >> %LOG%
    exit /b 1
)
echo [chain] %DATE% %TIME% self-test passed - starting full run >> %LOG%
python -u run_mission_eval.py --out_tag mission_eval_20261002 --out_root G:/mission_eval >> %LOG% 2>&1
echo [chain] %DATE% %TIME% full run exited with code %errorlevel% >> %LOG%
