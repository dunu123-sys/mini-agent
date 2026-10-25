@echo off
REM mini-agent web UI - portable launcher.
REM   run.cmd            start on the port from config.json (default 8790)
REM   set MINI_AGENT_ROOT first if state/ models/ logs/ live elsewhere
setlocal
set "HERE=%~dp0"
if not defined MINI_AGENT_ROOT set "MINI_AGENT_ROOT=%HERE%"
set "PY=python"
where py >nul 2>&1 && set "PY=py -3"
if not exist "%HERE%config.json" copy /y "%HERE%config.example.json" "%HERE%config.json" >nul
cd /d "%HERE%"
%PY% "%HERE%mini-agent-web.py"
endlocal
