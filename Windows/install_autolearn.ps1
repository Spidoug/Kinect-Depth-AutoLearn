param(
    [ValidateSet('CPU','NVIDIA','AMD')]
    [string]$Backend
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$sketchDir = Join-Path $root 'app\KinectDepthAutoLearn'
if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) { throw 'LOCALAPPDATA is not defined.' }
$persist = Join-Path $env:LOCALAPPDATA 'KinectDepthAutoLearn'
$runtimeDir = Join-Path $persist 'runtime'
$toolsDir = Join-Path $persist 'tools'
$cacheDir = Join-Path $persist 'cache'
$assetDir = Join-Path $persist 'assets'
$persistPointer = Join-Path $persist 'runtime_python.txt'
$persistStatus = Join-Path $persist 'runtime_status.json'
$localPointer = Join-Path $root 'runtime_python.txt'
$localStatus = Join-Path $root 'runtime_status.json'
$logFile = Join-Path $root 'install_autolearn.log'

New-Item -ItemType Directory -Force -Path $persist,$runtimeDir,$toolsDir,$cacheDir,$assetDir,(Join-Path $sketchDir 'code'),(Join-Path $sketchDir 'data') | Out-Null

function Say {
    param([string]$Text,[ConsoleColor]$Color=[ConsoleColor]::Gray)
    Write-Host $Text -ForegroundColor $Color
}
function Download-File {
    param([string]$Url,[string]$Out,[long]$MinBytes=1024)
    if ((Test-Path -LiteralPath $Out) -and ((Get-Item -LiteralPath $Out).Length -ge $MinBytes)) {
        Say "Using cached $(Split-Path -Leaf $Out)" DarkGreen
        return
    }
    $tmp = "$Out.part"
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    Say "Downloading $(Split-Path -Leaf $Out)..." Cyan
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $tmp -TimeoutSec 600
    if ((Get-Item -LiteralPath $tmp).Length -lt $MinBytes) { throw "Invalid download: $Url" }
    Move-Item -LiteralPath $tmp -Destination $Out -Force
}
function Ensure-Uv {
    $uv = Join-Path $toolsDir 'uv.exe'
    if (Test-Path -LiteralPath $uv) { return $uv }
    $zip = Join-Path $cacheDir 'uv-win64.zip'
    Download-File 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' $zip 100000
    $extract = Join-Path $cacheDir 'uv-extract'
    Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
    $found = Get-ChildItem -LiteralPath $extract -Filter uv.exe -Recurse -File | Select-Object -First 1
    if (-not $found) { throw 'uv.exe not found in downloaded archive.' }
    Copy-Item -LiteralPath $found.FullName -Destination $uv -Force
    return $uv
}
function Run-Native {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [string[]]$ArgumentList = @()
    )
    if (-not (Test-Path -LiteralPath $Exe)) { throw "Executable not found: $Exe" }
    $cmdText = "$Exe $($ArgumentList -join ' ')".Trim()
    Add-Content -LiteralPath $logFile -Value ("RUN: {0}" -f $cmdText) -Encoding UTF8
    & $Exe @ArgumentList
    $rc = $LASTEXITCODE
    if ($null -eq $rc) { $rc = 0 }
    if ($rc -ne 0) { throw "Command failed ($rc): $cmdText" }
}

function Get-Gpus {
    try { return @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name } | Where-Object { $_ }) }
    catch { return @() }
}

try {
    Set-Content -LiteralPath $logFile -Value ("Start: {0}" -f (Get-Date -Format o)) -Encoding UTF8
    if (-not $Backend) {
        Say 'Select training backend:' White
        Say '  [1] CPU'
        Say '  [2] NVIDIA/CUDA'
        Say '  [3] AMD/DirectML'
        $choice = Read-Host 'Option [1]'
        switch ($choice.Trim()) { '2' {$Backend='NVIDIA'} '3' {$Backend='AMD'} default {$Backend='CPU'} }
    }
    $selected = $Backend.ToUpperInvariant()
    $gpuText = ((Get-Gpus) -join ' | ')
    Say "Selected: $selected" Cyan
    Say "Adapters: $gpuText" DarkGray

    $uv = Ensure-Uv
    Run-Native -Exe $uv -ArgumentList @('--version')
    $env:UV_CACHE_DIR = Join-Path $cacheDir 'uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtimeDir 'python'
    $env:UV_LINK_MODE = 'copy'
    $env:UV_HTTP_TIMEOUT = '300'
    $env:UV_HTTP_RETRIES = '5'

    if ($selected -eq 'AMD') {
        # DirectML is deliberately pinned to Python 3.10 for the project runtime.
        # Microsoft currently documents a Python 3.10 environment for PyTorch + DirectML,
        # and the published torch-directml package provides a cp310 Windows wheel.
        $pythonVersion = '3.10'
        $meta = Invoke-RestMethod -Uri 'https://pypi.org/pypi/torch-directml/json' -TimeoutSec 90
        $directmlVersion = [string]$meta.info.version
        $releaseProp = $meta.releases.PSObject.Properties[$directmlVersion]
        if ($null -eq $releaseProp) { throw "torch-directml release metadata not found for $directmlVersion." }
        $hasCp310 = $false
        foreach ($f in @($releaseProp.Value)) {
            if (([string]$f.filename) -match 'cp310-cp310-win_amd64\.whl$') { $hasCp310 = $true; break }
        }
        if (-not $hasCp310) { throw "torch-directml $directmlVersion does not provide a Python 3.10 Windows wheel." }
        $backendName = 'DirectML'
        $venvDir = Join-Path $runtimeDir 'venv-directml'
    } elseif ($selected -eq 'NVIDIA') {
        $pythonVersion = if ($env:KDAL_PYTHON_VERSION) { $env:KDAL_PYTHON_VERSION } else { '3.12' }
        $backendName = 'CUDA'
        $venvDir = Join-Path $runtimeDir 'venv-cuda'
    } else {
        $pythonVersion = if ($env:KDAL_PYTHON_VERSION) { $env:KDAL_PYTHON_VERSION } else { '3.12' }
        $backendName = 'CPU'
        $venvDir = Join-Path $runtimeDir 'venv-cpu'
    }

    Run-Native -Exe $uv -ArgumentList @('python','install',$pythonVersion)
    $python = Join-Path $venvDir 'Scripts\python.exe'

    if (-not (Test-Path -LiteralPath $python)) { Run-Native -Exe $uv -ArgumentList @('venv','--python',$pythonVersion,$venvDir) }

    if ($selected -eq 'AMD') {
        Run-Native -Exe $uv -ArgumentList @('pip','install','--python',$python,"torch-directml==$directmlVersion")
        Run-Native -Exe $uv -ArgumentList @('pip','install','--python',$python,'numpy<2','pillow','onnx','onnxruntime','onnxscript')
    } elseif ($selected -eq 'NVIDIA') {
        $cudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { 'https://download.pytorch.org/whl/cu128' }
        Run-Native -Exe $uv -ArgumentList @('pip','install','--python',$python,'torch','--index-url',$cudaIndex)
        Run-Native -Exe $uv -ArgumentList @('pip','install','--python',$python,'numpy','pillow','onnx','onnxruntime','onnxscript')
    } else {
        Run-Native -Exe $uv -ArgumentList @('pip','install','--python',$python,'torch','--index-url','https://download.pytorch.org/whl/cpu')
        Run-Native -Exe $uv -ArgumentList @('pip','install','--python',$python,'numpy','pillow','onnx','onnxruntime','onnxscript')
    }

    [xml]$maven = (Invoke-WebRequest -UseBasicParsing -Uri 'https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/maven-metadata.xml' -TimeoutSec 60).Content
    $ortVersion = [string]$maven.metadata.versioning.release
    if ([string]::IsNullOrWhiteSpace($ortVersion)) { $ortVersion = '1.22.0' }
    $jarName = "onnxruntime-$ortVersion.jar"
    Download-File "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/$ortVersion/$jarName" (Join-Path $assetDir $jarName) 1000000
    Download-File 'https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/ed0f314bb7356fd1dbf1e4f52c2d40791bf6534f/onnx/model.onnx?download=true' (Join-Path $assetDir 'movenet.onnx') 100000
    Download-File 'https://huggingface.co/tasmulaev/rtmpose-m-distill/resolve/6fe5cf88e76efbef02648f341d75b10b8eb4acfa/onnx/rtmpose-m-distill-256x256.onnx?download=true' (Join-Path $assetDir 'rtmpose-hand.onnx') 100000

    $codeDir = Join-Path $sketchDir 'code'; $dataDir = Join-Path $sketchDir 'data'
    Get-ChildItem -LiteralPath $codeDir -Filter 'onnxruntime-*.jar' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Copy-Item -LiteralPath (Join-Path $assetDir $jarName) -Destination $codeDir -Force
    Copy-Item -LiteralPath (Join-Path $assetDir 'movenet.onnx') -Destination (Join-Path $dataDir 'movenet.onnx') -Force
    Copy-Item -LiteralPath (Join-Path $assetDir 'rtmpose-hand.onnx') -Destination (Join-Path $dataDir 'rtmpose-hand.onnx') -Force

    Set-Content -LiteralPath $persistPointer -Value $python -Encoding ASCII
    Set-Content -LiteralPath $localPointer -Value $python -Encoding ASCII
    $status = [ordered]@{version='1';platform='windows';backend=$backendName;selection=$selected;gpu=$gpuText;python=$python;python_version=$pythonVersion;persistent_root=$persist;installed=(Get-Date -Format o)}
    $status | ConvertTo-Json | Set-Content -LiteralPath $persistStatus -Encoding UTF8
    Copy-Item -LiteralPath $persistStatus -Destination $localStatus -Force

    Run-Native -Exe $python -ArgumentList @((Join-Path $root 'trainer\diagnose_runtime.py'))
    Say ''
    Say "[OK] Runtime ready: $backendName" Green
    Say "Persistent runtime: $persist" DarkGreen
    Say "Persistent cache: $cacheDir" DarkGreen
    Say 'Deleting/re-extracting the project will not delete downloaded dependencies.' DarkGreen
    exit 0
}
catch {
    $msg = $_.Exception.Message
    Add-Content -LiteralPath $logFile -Value ("ERROR: $msg") -Encoding UTF8
    Say ''
    Say 'ERROR - INSTALLATION STOPPED' Red
    Say $msg Red
    Say "Log: $logFile" Yellow
    exit 1
}
