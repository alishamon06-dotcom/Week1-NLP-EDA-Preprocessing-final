"""
Week 1 NLP EDA and preprocessing
Dataset: UCI SMS Spam Collection (5,574 messages)
Run: python src_eda.py
The script downloads the official UCI ZIP when the raw data file is absent.
All numerical summaries and charts in outputs/ are calculated from the downloaded file.
"""
from pathlib import Path
import io, zipfile, urllib.request, re, string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
RAW_FILE = DATA_DIR / "SMSSpamCollection"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def download_dataset():
    if RAW_FILE.exists():
        return
    print("Downloading the official UCI dataset...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next((n for n in archive.namelist()
                       if n.endswith("SMSSpamCollection")), None)
        if member is None:
            raise FileNotFoundError("SMSSpamCollection was not found in the ZIP.")
        RAW_FILE.write_bytes(archive.read(member))

def load_data():
    download_dataset()
    df = pd.read_csv(RAW_FILE, sep="\t", header=None,
                     names=["label", "text"], encoding="utf-8",
                     quoting=3, on_bad_lines="error")
    return df

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b\d+\b", " NUMBER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = load_data()
    print("\n--- DATASET OVERVIEW ---")
    print("Rows, columns:", df.shape)
    print("\nLabel counts:\n", df["label"].value_counts(dropna=False))
    print("\nMissing values:\n", df.isna().sum())
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    print("Duplicate message texts:", int(df["text"].duplicated().sum()))
    print("\nMessage length summary:\n", df["text"].astype(str).str.len().describe())

    # Validate labels and remove unusable records
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df = df[df["text"].str.strip().ne("")].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    print("Rows removed as duplicate label/text pairs:", before - len(df))

    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda x: len(wordpunct_tokenize(x)))
    df["clean_text"] = df["text"].apply(normalize_text)

    # Tokenization and stop-word removal (retain negation words)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english")) - {"no", "not", "nor"}
    df["tokens"] = df["clean_text"].apply(
        lambda x: [t for t in wordpunct_tokenize(x)
                   if t.isalpha() and t not in stop_words]
    )
    df["processed_text"] = df["tokens"].apply(lambda toks: " ".join(toks))

    # Label distribution
    counts = df["label"].value_counts()
    ax = counts.reindex(["ham", "spam"]).plot(kind="bar", title="SMS label distribution")
    ax.set_xlabel("Label"); ax.set_ylabel("Messages")
    plt.tight_layout(); plt.savefig(OUT_DIR / "label_distribution.png", dpi=180); plt.close()

    # Message length distribution
    for label in ["ham", "spam"]:
        subset = df.loc[df.label == label, "char_count"]
        plt.hist(subset, bins=35, alpha=0.55, label=label)
    plt.title("Message length by label"); plt.xlabel("Characters"); plt.ylabel("Frequency")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "message_length_by_label.png", dpi=180); plt.close()

    # Top unigrams and bigrams
    for ngram_range, label in [((1, 1), "unigrams"), ((2, 2), "bigrams")]:
        vec = CountVectorizer(ngram_range=ngram_range, stop_words="english",
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        matrix = vec.fit_transform(df["clean_text"])
        totals = matrix.sum(axis=0).A1
        terms = pd.DataFrame({"term": vec.get_feature_names_out(), "count": totals})
        terms = terms.sort_values("count", ascending=False).head(15)
        terms.to_csv(OUT_DIR / f"top_{label}.csv", index=False)
        plot_data = terms.sort_values("count")
        plt.figure(figsize=(8, 5))
        plt.barh(plot_data["term"], plot_data["count"])
        plt.title(f"Top 15 {label}"); plt.xlabel("Frequency")
        plt.tight_layout(); plt.savefig(OUT_DIR / f"top_{label}.png", dpi=180); plt.close()

    # Optional lexicon sentiment: descriptive only, not ground truth.
    # VADER is installed through vaderSentiment and may not capture SMS slang/irony.
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        df["sentiment_compound"] = df["text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"])
        df["sentiment_group"] = pd.cut(
            df["sentiment_compound"], bins=[-1.01, -0.05, 0.05, 1.01],
            labels=["Negative", "Neutral", "Positive"])
        sent_counts = df["sentiment_group"].value_counts().reindex(
            ["Negative", "Neutral", "Positive"], fill_value=0)
        sent_counts.to_csv(OUT_DIR / "sentiment_distribution.csv", header=["count"])
        sent_counts.plot(kind="bar", title="Lexicon-based sentiment (VADER)")
        plt.xlabel("Sentiment"); plt.ylabel("Messages")
        plt.tight_layout(); plt.savefig(OUT_DIR / "sentiment_distribution.png", dpi=180); plt.close()
    except ImportError:
        print("VADER not installed; optional sentiment analysis skipped.")

    # Save cleaned data and reproducible statistical summaries
    df.to_csv(OUT_DIR / "processed_sms.csv", index=False)
    summary = {
        "rows_after_cleaning": int(len(df)),
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "missing_values_after_cleaning": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_label_text_pairs_after_cleaning": int(df.duplicated(["label", "text"]).sum()),
        "mean_character_count": float(df["char_count"].mean()),
        "median_character_count": float(df["char_count"].median()),
        "mean_word_count": float(df["word_count"].mean()),
        "median_word_count": float(df["word_count"].median())
    }
    (OUT_DIR / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8")
    print("\n--- FINAL SUMMARY ---")
    print(__import__("json").dumps(summary, indent=2))
    print(f"\nCharts and tables saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
