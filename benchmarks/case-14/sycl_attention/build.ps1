$ErrorActionPreference = "Stop"

$oneApiRoot = "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0"
$compiler = Join-Path $oneApiRoot "bin\icx.exe"
$source = Join-Path $PSScriptRoot "sycl_attention.cpp"
$output = Join-Path $PSScriptRoot "sycl_attention.exe"
$libraryPath = Join-Path $oneApiRoot "lib"

if (-not (Test-Path -LiteralPath $compiler)) {
    throw "Intel oneAPI compiler not found at $compiler"
}

& $compiler -fsycl -O3 -Qstd=c++17 -EHsc $source -o $output `
    -link "-libpath:$libraryPath"
if ($LASTEXITCODE -ne 0) {
    throw "icx failed with exit code $LASTEXITCODE"
}

Write-Output "Built $output"

