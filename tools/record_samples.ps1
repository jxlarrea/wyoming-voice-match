<#
.SYNOPSIS
    Records voice enrollment samples for Wyoming Voice Match.
.DESCRIPTION
    Lists available microphones, lets you pick one, and records 
    WAV samples into the enrollment folder for a given speaker.
.PARAMETER Speaker
    Name of the speaker to enroll (e.g., "john").
.PARAMETER Samples
    Number of samples to record (default: 7).
.PARAMETER Duration
    Duration of each sample in seconds (default: 5).
.EXAMPLE
    .\record_samples.ps1 -Speaker john
    .\record_samples.ps1 -Speaker jane -Samples 10 -Duration 8
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Speaker,

    [int]$Samples = 7,

    [int]$Duration = 5
)

$ErrorActionPreference = "Stop"

# Check ffmpeg is installed
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found. Install it with: winget install ffmpeg" -ForegroundColor Red
    exit 1
}

# Get audio devices from ffmpeg
Write-Host "`nDetecting audio devices..." -ForegroundColor Cyan
$rawOutput = & cmd /c "ffmpeg -sources dshow 2>&1"

# Parse audio device names and paths
$devices = @()
$devicePaths = @()
$lines = if ($rawOutput -is [array]) { $rawOutput } else { $rawOutput -split "`n" }
foreach ($line in $lines) {
    $lineStr = "$line"
    if ($lineStr -match '^\s+(\S+)\s+\[(.+?)\]\s*\(audio\)') {
        $devicePaths += $Matches[1]
        $devices += $Matches[2]
    }
}

# Fallback: try legacy command if no devices found
if ($devices.Count -eq 0) {
    $rawOutput = & cmd /c "ffmpeg -list_devices true -f dshow -i dummy 2>&1"
    $inAudio = $false
    $lines = if ($rawOutput -is [array]) { $rawOutput } else { $rawOutput -split "`n" }
    foreach ($line in $lines) {
        $lineStr = "$line"
        if ($lineStr -match 'DirectShow audio devices') { $inAudio = $true; continue }
        if ($lineStr -match 'DirectShow video devices') { $inAudio = $false; continue }
        if ($inAudio -and $lineStr -match '"(.+)"') {
            $devices += $Matches[1]
            $devicePaths += $Matches[1]
        }
    }
}

if ($devices.Count -eq 0) {
    Write-Host "No audio devices found. Make sure a microphone is connected." -ForegroundColor Red
    exit 1
}

# Display device list
Write-Host "`nAvailable microphones:" -ForegroundColor Green
for ($i = 0; $i -lt $devices.Count; $i++) {
    Write-Host "  [$($i + 1)] $($devices[$i])"
}

# User selection
do {
    $selection = Read-Host "`nSelect a microphone (1-$($devices.Count))"
} while (-not ($selection -as [int]) -or [int]$selection -lt 1 -or [int]$selection -gt $devices.Count)

$micName = $devices[[int]$selection - 1]
$micPath = $devicePaths[[int]$selection - 1]
Write-Host "`nUsing: $micName" -ForegroundColor Green

# Create enrollment directory relative to script location (tools/../data/enrollment)
$enrollDir = Join-Path $PSScriptRoot "..\data\enrollment\$Speaker"
if (-not (Test-Path $enrollDir)) {
    New-Item -ItemType Directory -Path $enrollDir -Force | Out-Null
}

Write-Host "`nRecording $Samples samples ($Duration seconds each) for speaker '$Speaker'" -ForegroundColor Cyan
Write-Host "Speak naturally - vary your volume and distance from the mic.`n"

$phrases = @(
    "Hey, turn on the living room lights and set them to fifty percent",
    "What is the weather going to be like tomorrow morning",
    "Set a timer for ten minutes and remind me to check the oven",
    "Play some jazz music in the kitchen please",
    "Good morning, what is on my calendar for today",
    "Lock the front door and turn off all the lights downstairs",
    "What is the temperature inside the house right now",
    "Turn the thermostat up to seventy two degrees",
    "Add milk and eggs to my shopping list",
    "Dim the bedroom lights to twenty percent"
)

for ($i = 1; $i -le $Samples; $i++) {
    $outFile = Join-Path $enrollDir "sample_$i.wav"
    $phrase = $phrases[($i - 1) % $phrases.Count]

    Write-Host "[$i/$Samples] Say: `"$phrase`"" -ForegroundColor Yellow
    Write-Host "  Recording in 2 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 2

    Write-Host "  Recording..." -ForegroundColor Red
    & ffmpeg -y -f dshow -i "audio=$micPath" -ar 16000 -ac 1 -t $Duration $outFile -loglevel quiet 2>$null

    if (Test-Path $outFile) {
        Write-Host "  Saved: $outFile" -ForegroundColor Green
    } else {
        Write-Host "  Failed to record sample $i" -ForegroundColor Red
    }

    if ($i -lt $Samples) {
        Start-Sleep -Seconds 1
    }
}

Write-Host "`nDone! Recorded $Samples samples in: $enrollDir" -ForegroundColor Cyan
Write-Host "Now run enrollment to generate the voiceprint:" -ForegroundColor Cyan
Write-Host "  docker compose run --rm --entrypoint python voice-match -m scripts.enroll --speaker $Speaker" -ForegroundColor White
