from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_charts.chart_features import compare_two_corpora_and_plot_grouped
from report_table_latex import auc_report_to_latex
import pandas as pd



# -----------------------------
# RUN WITH "PLAY"
# -----------------------------
if __name__ == "__main__":
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name("mingotezaragoza")
    corpusA = Corpus(name=corpus.name, subcorpus="profano", id=None)
    corpusB = Corpus(name=corpus.name, subcorpus="religioso", id=None)

    CORPUS_A_DIR = resolver.data_clean(corpusA, corpusA.subcorpus)
    CORPUS_B_DIR = resolver.data_clean(corpusB, corpusB.subcorpus)

    LABEL_A = "Profano"
    LABEL_B = "Religioso"

    # Si lo dejas en None => abre varias ventanas (una por categoría)
    # Si pones carpeta => guarda varios PNG en esa carpeta
    OUT_DIR =  resolver.charts_dir() / "apps_teimus" / corpus.name
    out_csv_a = OUT_DIR / f"features_Profano_{corpus.name}.csv"
    out_csv_b = OUT_DIR/ f"features_Religioso_{corpus.name}.csv"

    dfA, dfB = compare_two_corpora_and_plot_grouped(
        corpus_a_dir=str(CORPUS_A_DIR),
        corpus_b_dir=str(CORPUS_B_DIR),
        label_a=LABEL_A,
        label_b=LABEL_B,
        bins=30,
        out_dir=OUT_DIR,
        save_csv=True,
        out_csv_a=out_csv_a,
        out_csv_b=out_csv_b
    )


    #MUESTRA TABLAS AUC POR FEATURE
    df_prof = pd.read_csv(out_csv_a)
    df_reli = pd.read_csv(out_csv_b)

    # Si tienes columnas tipo 'path', 'file', 'id', etc., exclúyelas:
    exclude = {"path", "filepath", "file", "filename"}

    ranking = auc_report_to_latex(
        df_a=df_prof,
        df_b=df_reli,
        label_a="Profano",
        label_b="Religioso",
        exclude_cols=exclude,
        top_k=12,
    )