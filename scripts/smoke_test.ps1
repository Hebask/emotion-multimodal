Write-Host "== Smoke test: Text ==" -ForegroundColor Cyan
python -m src.app --text "I am very happy today!"

Write-Host "`n== Smoke test: Image (sample) ==" -ForegroundColor Cyan
python -m src.app --image ".\samples\face.jpg"

Write-Host "`n== Smoke test: Audio (sample) ==" -ForegroundColor Cyan
python -m src.app --audio ".\samples\speech.wav"

Write-Host "`n== Smoke test: Fusion (text+image+audio) ==" -ForegroundColor Cyan
python -m src.app --text "I feel sad and tired." --image ".\samples\face.jpg" --audio ".\samples\speech.wav"

Write-Host "`n== Done ==" -ForegroundColor Green
