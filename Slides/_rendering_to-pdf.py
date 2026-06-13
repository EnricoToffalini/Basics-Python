from pathlib import Path
import subprocess

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

for html in Path(".").glob("*.html"):
    pdf = html.with_suffix(".pdf")
    url = html.resolve().as_uri() + "?print-pdf"

    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--print-to-pdf=" + str(pdf.resolve()),
            "--print-to-pdf-no-header",
            url
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"Creato: {pdf}")
