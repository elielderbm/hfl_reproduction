# Análises

- `extract_metrics.py`: consolida logs JSONL → `outputs/metrics_all.csv`
- `plot_curves.py`: gera gráficos (acurácia/perda local, janela/p, edges ao longo de rounds)
- `summarize.py`: gera sumários em CSV

Exemplo:
```bash
docker compose run --rm analyzer python -m project.analysis.extract_metrics
docker compose run --rm analyzer python -m project.analysis.plot_curves
docker compose run --rm analyzer python -m project.analysis.summarize
```
