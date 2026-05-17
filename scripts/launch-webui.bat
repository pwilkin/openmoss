@echo off
REM Launch moss-tts-server.exe with the bundled WebUI.
REM
REM Usage:
REM   scripts\launch-webui.bat <model.gguf> [extra moss-tts-server args...]
REM
REM Environment overrides:
REM   MOSS_SERVER  path to moss-tts-server.exe (default: auto-detect)
REM   MOSS_HOST    host to bind                (default: 127.0.0.1)
REM   MOSS_PORT    port to bind                (default: 8080)
REM   MOSS_WEBUI   path to webui directory     (default: auto-detect)

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo usage: %~nx0 ^<model.gguf^> [extra moss-tts-server args...] 1>&2
    exit /b 2
)

set "MODEL=%~1"
shift

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI"

if not defined MOSS_SERVER (
    for %%C in (
        "%PROJECT_DIR%\build\Release\moss-tts-server.exe"
        "%PROJECT_DIR%\build\moss-tts-server.exe"
        "%PROJECT_DIR%\moss-tts-server.exe"
    ) do (
        if exist "%%~C" (
            set "MOSS_SERVER=%%~C"
            goto :have_server
        )
    )
)
:have_server
if not defined MOSS_SERVER (
    echo error: moss-tts-server.exe not found; set MOSS_SERVER or build the project first 1>&2
    exit /b 1
)

if not defined MOSS_WEBUI (
    for %%C in (
        "%PROJECT_DIR%\webui"
    ) do (
        if exist "%%~C\index.html" (
            set "MOSS_WEBUI=%%~C"
            goto :have_webui
        )
    )
    for %%D in ("%MOSS_SERVER%") do (
        if exist "%%~dpDwebui\index.html" set "MOSS_WEBUI=%%~dpDwebui"
    )
)
:have_webui
if not defined MOSS_WEBUI (
    echo error: webui directory not found; set MOSS_WEBUI 1>&2
    exit /b 1
)

if not defined MOSS_HOST set "MOSS_HOST=127.0.0.1"
if not defined MOSS_PORT set "MOSS_PORT=8080"

echo ----------------------------------------------------
echo  openmoss TTS - WebUI
echo    binary : %MOSS_SERVER%
echo    model  : %MODEL%
echo    webui  : %MOSS_WEBUI%
echo    open   : http://%MOSS_HOST%:%MOSS_PORT%/
echo ----------------------------------------------------

"%MOSS_SERVER%" --model "%MODEL%" --host "%MOSS_HOST%" --port "%MOSS_PORT%" --webui-dir "%MOSS_WEBUI%" %*
