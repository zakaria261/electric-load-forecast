from pathlib import Path
import zipfile
import requests

URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

ZIP_PATH = DATA_DIR / "household_power_consumption.zip"
DATA_PATH = DATA_DIR / "household_power_consumption.txt"


def download_data():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print("Dataset déjà présent.")
        return

    print("Téléchargement du dataset UCI...")

    with requests.get(URL, stream=True, timeout=60) as response:
        response.raise_for_status()

        with open(ZIP_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print("Extraction du dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extract(
            "household_power_consumption.txt",
            DATA_DIR
        )

    ZIP_PATH.unlink()

    print("Téléchargement terminé.")
    print(f"Fichier : {DATA_PATH}")


if __name__ == "__main__":
    download_data()