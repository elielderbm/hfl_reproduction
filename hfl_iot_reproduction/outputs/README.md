Relatorios e graficos gerados pelos scripts em `project/analysis/`.

Arquivos principais esperados:
- `metrics_all.csv`: consolidacao de `logs/*.jsonl`.
- `iot_summary.csv`, `edge_summary.csv`, `cloud_summary.csv`: sumarios tabulares.
- `results_explanation.md`: explicacao narrativa contextualizada no app IoT->Edge->Cloud.
- `*.png`: curvas produzidas por `plot_curves.py` (ex.: `iot_*_acc.png`, `iot_*_loss.png`).
- `paper_*.csv` e `paper_plots/*.png`: tabelas e gráficos alinhados ao artigo (round time, throughput, overhead, global acc).
- `paper_report.md`: relatório dinâmico alinhado ao paper (inclui tendências e comparação async vs sync).
