@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto fail
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m streamlit run app.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo Nao foi possivel iniciar. Confira Python 3.11+ e as dependencias no README.
pause
exit /b 1
