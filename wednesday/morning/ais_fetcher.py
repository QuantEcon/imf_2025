import os
import re

import bs4
import requests

marine_cadastre_base_url = (
    "https://coast.noaa.gov" + "/htdata/CMSP/AISDataHandler" + "/{year}/"
)
ais_file_matcher = re.compile(
    r"^ais[-_]\d{4}[-_]\d{2}[-_]\d{2}\.[a-z]+", re.IGNORECASE
)


def fetch_ais_data(year):
    year_base_url = marine_cadastre_base_url.format(year=year)

    # Fetch the year's index page
    index_url = year_base_url + "index.html"
    index_res = requests.get(index_url)
    index_page = index_res.text

    # Turn index page into soup
    soupy_soup = bs4.BeautifulSoup(index_page)

    links_to_download = []
    for link in soupy_soup.find_all("a"):
        if ais_file_matcher.match(link.text):
            filename = link.text
            download_link = year_base_url + link.attrs["href"]

            print(f"\tFound match: {filename}")
            print(f"\tDownloading: {download_link}")
            res = requests.get(download_link)

            with open(f"wednesday/morning/data/{year}/{filename}", "wb") as f:
                f.write(res.content)

    return True


if __name__ == "__main__":
    os.makedirs("wednesday/morning/data", exist_ok=True)

    for year in range(2025, 2023, -1):
        print(f"Working on the year {year}")
        os.makedirs(f"wednesday/morning/data/{year}", exist_ok=True)

        success = fetch_ais_data(year)
        print(f"{success} for {year}")
