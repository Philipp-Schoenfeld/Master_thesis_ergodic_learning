$env:PYTHONPATH = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\thesis_architecture;c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\src"

$python = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\.venv\Scripts\python.exe"
$script = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\thesis_architecture\visualize_checkpoint.py"
$ckpt = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\thesis_architecture\checkpoints\netz2d_laenge.pt"
$out = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\thesis_architecture\visualisierungsordner\2d_laenge"

if (-not (Test-Path $out)) {
    New-Item -ItemType Directory -Force -Path $out
}

# Ziellängen, z.B. 2, 4, 6, 8, 10
$lengths = @(2.0, 4.0, 6.0, 8.0, 10.0)

# Set encoding to avoid any remaining unicode issues
$env:PYTHONIOENCODING = "utf-8"

foreach ($l in $lengths) {
    Write-Host "Generiere mit Ziellänge $l ..."
    & $python $script --checkpoint $ckpt --out_dir $out --length $l --length_cfg_weight 2.0 --n_gen 5 --obstacle_mode off
}
Write-Host "Fertig! Ergebnisse liegen in $out"
