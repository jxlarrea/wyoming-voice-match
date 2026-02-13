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

    [int]$Samples = 30,

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
    "Dim the bedroom lights to twenty percent",
    "What time is my first meeting tomorrow",
    "Turn off the TV in the living room",
    "Set an alarm for seven thirty in the morning",
    "Open the garage door",
    "Tell me a joke",
    "How long is my commute to work today",
    "Play my morning playlist on the bedroom speaker",
    "Is the back door locked",
    "Remind me to call the dentist at noon",
    "What is the humidity outside right now",
    "Turn on the fan in the office",
    "Cancel all my alarms for tomorrow",
    "Start the robot vacuum in the living room",
    "How many steps have I taken today",
    "Read me the latest news headlines",
    "Set the lights to warm white in the dining room",
    "Is there any rain expected this weekend",
    "Pause the music for a moment please",
    "Turn on do not disturb mode",
    "Show me the front door camera"
)

# Find existing samples
$existing = Get-ChildItem -Path $enrollDir -Filter "*.wav" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Found $($existing.Count) existing sample(s) in folder. New samples will be added." -ForegroundColor Cyan
}

for ($i = 0; $i -lt $Samples; $i++) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outFile = Join-Path $enrollDir "${Speaker}_${timestamp}.wav"
    $phrase = $phrases[$i % $phrases.Count]

    Write-Host "[$($i + 1)/$Samples] Say: `"$phrase`"" -ForegroundColor Yellow
    Write-Host "  Recording in 2 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 2

    Write-Host "  Recording..." -ForegroundColor Red
    & ffmpeg -y -f dshow -i "audio=$micPath" -ar 16000 -ac 1 -t $Duration $outFile -loglevel quiet 2>$null

    if (Test-Path $outFile) {
        Write-Host "  Saved: $outFile" -ForegroundColor Green
    } else {
        Write-Host "  Failed to record sample" -ForegroundColor Red
    }

    if ($i -lt ($Samples - 1)) {
        Start-Sleep -Seconds 1
    }
}

Write-Host "`nDone! Recorded $Samples samples in: $enrollDir" -ForegroundColor Cyan
Write-Host "Now run enrollment to generate the voiceprint:" -ForegroundColor Cyan
Write-Host "  docker compose run --rm wyoming-voice-match python -m scripts.enroll --speaker $Speaker" -ForegroundColor White