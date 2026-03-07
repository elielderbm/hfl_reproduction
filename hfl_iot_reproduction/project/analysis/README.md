# Análises

- `extract_metrics.py`: consolida logs JSONL → `outputs/metrics_all.csv`
- `plot_curves.py`: gera gráficos (score/erro local, **R2/MAPE** para regressão, **acurácia/BCE** para classificação, **distill RMSE**, janela/p do **cloud** e métricas globais por target)
- `summarize.py`: gera sumários em CSV (inclui `cloud_agg_summary.csv`) e `outputs/results_explanation.md`
- `explain_results.py`: gera apenas a explicação narrativa (separadamente)
- `paper_metrics.py`: gera tabelas/gráficos alinhados ao artigo (round time, throughput, overhead, **global RMSE/score/R2/MAPE/ACC** por target; para classificação, score=acurácia e RMSE=Brier)
- `paper_report.py`: gera relatório dinâmico alinhado ao paper (inclui análise temporal e comparação entre execuções)
- `convergence_report.py`: gera relatório focado em **convergência e erro por round** (score/RMSE/R2/MAPE; para classificação, score=acurácia)
- `target_report.py`: avalia **modelo global por target** (temp vs humidity vs light) usando pesos salvos
- `compare_async_sync.py`: relatório profundo de comparação **entre duas execuções** (Run A vs Run B, com métricas locais/globais)

Exemplo:
```bash
sudo docker compose run --rm analyzer python -m project.analysis.extract_metrics
sudo docker compose run --rm analyzer python -m project.analysis.plot_curves
sudo docker compose run --rm analyzer python -m project.analysis.summarize
sudo docker compose run --rm analyzer python -m project.analysis.paper_metrics
sudo docker compose run --rm analyzer python -m project.analysis.paper_report
# convergência por round
sudo docker compose run --rm analyzer python -m project.analysis.convergence_report --window 5 --show-rounds 5
# análise por target (requer SAVE_GLOBAL_WEIGHTS=1 na execução do cloud)
sudo docker compose run --rm analyzer python -m project.analysis.target_report --every 1
# comparação entre execuções
sudo docker compose run --rm analyzer python -m project.analysis.compare_async_sync --async-dir outputs/experiments/<RUN_ID>/run_a --sync-dir outputs/experiments/<RUN_ID>/run_b
# opcional (já é chamado dentro de summarize)
sudo docker compose run --rm analyzer python -m project.analysis.explain_results
```
