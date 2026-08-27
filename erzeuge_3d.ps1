$env:PYTHONPATH = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\3D_ergodic_learning;c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\thesis_architecture;c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\src"
$env:PYTHONIOENCODING = "utf-8"

$python = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\.venv\Scripts\python.exe"
$script = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\3D_ergodic_learning\run_surface_eval.py"
$ckpt = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\thesis_architecture\transfer\netz3d_flaechen.pt"
$out = "c:\Users\Philipp\Documents\Uni\Master_Thesis\Master_thesis_ergodic_learning\3D_ergodic_learning\results\surfaces"

Write-Host "Starte 3D Inferenz auf Oberflächen (Surfaces)..."
& $python $script --ckpt $ckpt --out_dir $out --shapes 5
Write-Host "Fertig! Ergebnisse liegen in $out"
