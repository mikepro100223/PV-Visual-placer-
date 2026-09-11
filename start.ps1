# Self-bootstrapping launcher: installs whatever is missing, then starts the app.
# Safe to run repeatedly; setup steps are skipped once they are done.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location -LiteralPath $PSScriptRoot

$Root = $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'
$Web = Join-Path $Root 'web'
$TorchVersion = '2.10.0'
$VisionVersion = '0.25.0'
$MinNode = [version]'22.13.0'

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Test-Command($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

function Install-WithWinget($id, $label) {
    if (-not (Test-Command 'winget')) {
        throw "$label is not installed and winget is unavailable. Install $label manually, then run start.cmd again."
    }
    Step "Installing $label via winget (may ask for permission)"
    & winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

function File-Hash($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }

# ---------------------------------------------------------------- Python
# Returns a command array for a working Python 3.10-3.14, or $null.
function Find-Python {
    $candidates = @(
        @('py', '-3.14'), @('py', '-3.13'), @('py', '-3.12'), @('py', '-3.11'), @('py', '-3.10'),
        @('python'), @('python3')
    )
    foreach ($c in $candidates) {
        if (-not (Test-Command $c[0])) { continue }
        $extra = @($c | Select-Object -Skip 1)
        try {
            # Also filters out the Microsoft Store "python.exe" stub, which prints nothing useful.
            # No double quotes in the code: PowerShell 5.1 strips them from native arguments.
            $v = & $c[0] @extra -c 'import sys; print(*sys.version_info[:2], sep=chr(46))' 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) {
                $ver = [version]($v | Select-Object -Last 1).Trim()
                if ($ver -ge [version]'3.10' -and $ver -lt [version]'3.15') { return ,$c }
            }
        } catch { }
    }
    return $null
}

function Test-VenvHealthy {
    if (-not (Test-Path -LiteralPath $VenvPy)) { return $false }
    try { & $VenvPy -c 'import sys' 2>$null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

if (-not (Test-VenvHealthy)) {
    if (Test-Path -LiteralPath $Venv) {
        Step 'Existing .venv is broken or from another machine; recreating it'
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }
    $py = Find-Python
    if (-not $py) {
        Install-WithWinget 'Python.Python.3.12' 'Python 3.12'
        $py = Find-Python
        if (-not $py) { throw 'Python 3.10-3.14 not found after install. Install Python 3.12 from python.org, then run start.cmd again.' }
    }
    Step "Creating virtual environment with: $($py -join ' ')"
    $pyArgs = @($py | Select-Object -Skip 1) + @('-m', 'venv', $Venv)
    Invoke-Checked $py[0] $pyArgs
}

# ---------------------------------------------------------------- Python packages
$ReqFile = Join-Path $Root 'requirements.txt'
$PyStamp = Join-Path $Venv '.deps-installed'
$hasGpu = [bool](Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue)
$flavor = if ($hasGpu) { 'cu128' } else { 'cpu' }
$wantPyStamp = "$(File-Hash $ReqFile)|torch=$TorchVersion+$flavor"
$havePyStamp = if (Test-Path -LiteralPath $PyStamp) { (Get-Content -LiteralPath $PyStamp -Raw).Trim() } else { '' }

if ($havePyStamp -ne $wantPyStamp) {
    Step 'Installing Python packages (first run can take several minutes)'
    Invoke-Checked $VenvPy @('-m', 'pip', 'install', '--upgrade', 'pip')
    Step "Installing PyTorch ($flavor build)"
    Invoke-Checked $VenvPy @('-m', 'pip', 'install',
        "torch==$TorchVersion", "torchvision==$VisionVersion",
        '--index-url', "https://download.pytorch.org/whl/$flavor")
    Invoke-Checked $VenvPy @('-m', 'pip', 'install', '-r', $ReqFile)
    Set-Content -LiteralPath $PyStamp -Value $wantPyStamp -Encoding ascii
}

# ---------------------------------------------------------------- Node.js
function Get-NodeVersion {
    if (-not (Test-Command 'node')) { return $null }
    try { return [version]((& node --version).Trim().TrimStart('v')) } catch { return $null }
}

$nodeVer = Get-NodeVersion
if (-not $nodeVer -or $nodeVer -lt $MinNode) {
    Install-WithWinget 'OpenJS.NodeJS.LTS' "Node.js $MinNode or newer"
    $nodeVer = Get-NodeVersion
    if (-not $nodeVer -or $nodeVer -lt $MinNode) {
        throw "Node.js $MinNode or newer is required (found: $nodeVer). Install the LTS from nodejs.org, then run start.cmd again."
    }
}

# ---------------------------------------------------------------- Frontend packages
$Lock = Join-Path $Web 'package-lock.json'
$NodeModules = Join-Path $Web 'node_modules'
$WebStamp = Join-Path $NodeModules '.deps-installed'
$wantWebStamp = "$(File-Hash $Lock)|node=$nodeVer"
$haveWebStamp = if (Test-Path -LiteralPath $WebStamp) { (Get-Content -LiteralPath $WebStamp -Raw).Trim() } else { '' }

if ($haveWebStamp -ne $wantWebStamp) {
    Step 'Installing frontend packages'
    Push-Location -LiteralPath $Web
    try { Invoke-Checked 'npm.cmd' @('ci', '--no-audit', '--no-fund') } finally { Pop-Location }
    Set-Content -LiteralPath $WebStamp -Value $wantWebStamp -Encoding ascii
}

# ---------------------------------------------------------------- Run
Step 'Starting PV Visual Placer'
& $VenvPy (Join-Path $Root 'scripts\serve.py') --open
exit $LASTEXITCODE
