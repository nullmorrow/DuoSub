# Create DuoSub.lnk next to DuoSub.pyw with the DuoSub icon (a .pyw file itself always shows Python's icon).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File tools\make_shortcut.ps1
$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $env:WINDIR "pyw.exe"          # the same launcher a double-click on DuoSub.pyw uses
if (-not (Test-Path $launcher)) { $launcher = (Get-Command pythonw.exe -ErrorAction Stop).Source }
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path $root "DuoSub.lnk"))
$link.TargetPath = $launcher
$link.Arguments = '"' + (Join-Path $root "DuoSub.pyw") + '"'
$link.WorkingDirectory = $root
$link.IconLocation = (Join-Path $root "assets\duosub.ico") + ",0"
$link.Description = "双语字幕工坊 DuoSub"
$link.Save()
Write-Output "created $(Join-Path $root 'DuoSub.lnk')"
