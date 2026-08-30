$ErrorActionPreference = "Stop"

$oneApiBin = "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\bin"
$executable = Join-Path $PSScriptRoot "sycl_attention.exe"

if (-not (Test-Path -LiteralPath $executable)) {
    throw "Build first with benchmarks\case-14\sycl_attention\build.ps1"
}

$env:Path = $oneApiBin + ";" + $env:Path
& $executable @args
exit $LASTEXITCODE

