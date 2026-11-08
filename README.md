# moss-image-ryzen-ai-npu (draft)

```powershell
pwsh -NoProfile -File .\setup.ps1 -Size 512
pwsh -NoProfile -File .\doctor.ps1 -Size 512
pwsh -NoProfile -File .\gen.ps1 -Items .\examples\one.json -Out .\outputs\first -Size 512
pwsh -NoProfile -File .\gen.ps1 -Items .\examples\repro_512.json -Out .\outputs\repro512 -Size 512 -Vae cpu
pwsh -NoProfile -File .\gen.ps1 -Items .\examples\repro_1024.json -Out .\outputs\repro1024 -Size 1024 -Vae npu
```

```powershell
pwsh -NoProfile -File .\setup.ps1 -Size all -NpuEnv mossnpu-p7-npu -TextEnv mossnpu-p7-text
pwsh -NoProfile -File .\doctor.ps1 -Size 1024
```

```powershell
# rebuild parts (convert)
& <text-python> -u .\tools\convert\convert_all.py --help
```
