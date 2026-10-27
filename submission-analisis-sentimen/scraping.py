"""Scraping ulasan aplikasi Halodoc di Google Play Store.

Jalankan:
    python scraping.py --count 20000 --output dataset_sentimen.csv

Label sentimen diturunkan dari rating bintang:
    1-2 = negatif, 3 = netral, 4-5 = positif
"""

import argparse
import time

import pandas as pd
from google_play_scraper import Sort, reviews


def label_sentiment(score: int) -> str:
    """Ubah rating bintang menjadi kelas sentimen."""
    if score <= 2:
        return "negatif"
    if score == 3:
        return "netral"
    return "positif"


def scrape_reviews(
    app_id: str = "com.linkdokter.halodoc.android",
    target_count: int = 20000,
    lang: str = "id",
    country: str = "id",
    batch_size: int = 2000,
    delay: float = 0.5,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Ambil ulasan terbaru sampai target_count atau data habis."""
    collected = []
    token = None

    while len(collected) < target_count:
        batch = min(batch_size, target_count - len(collected))
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result, token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=batch,
                    continuation_token=token,
                )
                break
            except Exception as exc:  # koneksi/rate limit sementara
                print(f"  percobaan {attempt}/{max_retries} gagal: {exc}")
                time.sleep(2 * attempt)

        if not result:
            print("Tidak ada ulasan baru; proses scraping berhenti.")
            break

        collected.extend(result)
        print(f"  {len(collected):,} ulasan terkumpul")
        if token is None:
            break
        time.sleep(delay)

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected).drop_duplicates(subset=["reviewId"])
    return df


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Pilih kolom yang dipakai, buang ulasan kosong, dan beri label."""
    if raw.empty:
        return raw

    df = raw[["reviewId", "score", "content", "at"]].rename(
        columns={"content": "text"}
    )
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["reviewId"]).reset_index(drop=True)
    df["sentiment"] = df["score"].apply(label_sentiment)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ulasan Google Play")
    parser.add_argument("--app-id", default="com.linkdokter.halodoc.android")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--output", default="dataset_sentimen.csv")
    parser.add_argument("--lang", default="id")
    parser.add_argument("--country", default="id")
    args = parser.parse_args()

    print(f"Mulai scraping {args.count:,} ulasan untuk {args.app_id} ...")
    raw = scrape_reviews(
        app_id=args.app_id,
        target_count=args.count,
        lang=args.lang,
        country=args.country,
    )
    df = build_dataset(raw)
    if df.empty:
        raise SystemExit("Scraping gagal: tidak ada ulasan yang diperoleh.")

    df.to_csv(args.output, index=False)
    print(f"\nTersimpan: {args.output} ({len(df):,} baris)")
    print(df["sentiment"].value_counts().to_string())
    print("\nRentang tanggal:", df["at"].min(), "s/d", df["at"].max())

    if len(df) < 3000:
        raise SystemExit("Jumlah data kurang dari 3.000 sampel.")


if __name__ == "__main__":
    main()
