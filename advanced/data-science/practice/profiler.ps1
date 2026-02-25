$models = @("knn", "svc", "svc_maxabs", "logreg_std")

foreach ($model in $models) {
    Write-Host "Profiling $model..."
    C:\IntelSDE\sde.exe -mix -top_blocks 0 -omix "out\${model}.txt" -- python models.py $model
}