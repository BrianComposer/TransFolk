INTEGRACION: DISTRIBUCIONES PROFANO VS RELIGIOSO + KS/BONFERRONI
=================================================================

Archivos incluidos:

1) feature_distributions.py
   Copiar en:
      transfolk_classifier/feature_distributions.py

2) run_classifier.py
   Version integrada con el modo final `distributions`.
   Copiar en:
      apps_teimus/run_classifier.py

Uso recomendado:

   python -m apps_teimus.run_classifier <root> all distributions

Pipeline completo:

   python -m apps_teimus.run_classifier <root> all train,eval,compare,features,curves,ablation,interpretability,distributions,final

Salida:

   experiments/teimus/classifiers/<corpus>/final_results/feature_distributions/

CSV principales:

   final_feature_distribution_values_all_seeds.csv
      Valores de features agregados desde todas las seeds.

   final_feature_distribution_stats_by_seed.csv
      Estadisticos Profano vs Religioso por seed.

   final_feature_distribution_stats_mean_std.csv
      Estadisticos agregados multi-seed, efecto Cohen's d, KS y Bonferroni.

   final_feature_distribution_ks_bonferroni.csv
      Test Kolmogorov-Smirnov de dos muestras por feature, p-value original,
      p-value corregido por Bonferroni, umbral alpha/m y significacion.

   final_feature_distribution_curves.csv
      Curvas de densidad medias y desviaciones entre seeds.

Figuras:

   final_feature_distribution_effects_top_<N>.png/.eps
      Ranking de diferencias Profano - Religioso por Cohen's d. Las features
      significativas tras Bonferroni aparecen con borde negro y asteriscos.

   final_distribution_<feature>_profano_vs_religioso.png/.eps
      Distribucion agregada multi-seed de cada feature. Cada figura incluye
      KS D, p-value corregido por Bonferroni y asteriscos de significacion.

Nota metodologica:

   El test KS se calcula sobre piezas unicas por __path, no sobre las copias
   repetidas al agregar varias seeds. Esto evita pseudorreplicacion, ya que los
   valores de las features dependen de la pieza y no de la seed. Las seeds se
   usan para estimar la estabilidad visual de las distribuciones.

