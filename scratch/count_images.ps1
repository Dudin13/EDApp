$path = 'data/datasets/veo_frames_raw/images/'
$imgs = Get-ChildItem -Path $path -File
Write-Host "Total images: $($imgs.Count)"
$prefixes = $imgs.Name | ForEach-Object { $_ -replace '_min\d+.*|_f\d+.*', '' } | Select-Object -Unique | Sort-Object
Write-Host "Unique match prefixes:"
foreach ($p in $prefixes) { Write-Host $p }
