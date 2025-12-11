#! pip install selenium webdriver-manager

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from pathlib import Path


# --- SETTINGS ---

CITIES = [
    "New York",
    "Austin",
    "Houston",
    "Cleveland",
    "Miami",
    "Boston",
    "Chicago",
    "Los Angeles",
    "San Francisco",
    "Seattle",
]

# URL
TOOL_URL = "https://www.omkar.cloud/tools/tripadvisor-scraper"  

# Folder where downloaded JSON files will be saved
DOWNLOAD_DIR = Path.cwd() / "tripadvisor_downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)


def create_driver():
    chrome_options = Options()
    chrome_options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"  # Chrome path

    prefs = {
        "download.default_directory": str(DOWNLOAD_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # specify the path of the chromedriver 
    service = Service(r"C:\tools\chromedriver_142\chromedriver.exe")

    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.maximize_window()
    return driver


def wait_for_download(before_files, timeout=120):
    """Finds and returns the newly downloaded file."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        current_files = set(DOWNLOAD_DIR.glob("*"))
        new_files = current_files - before_files
        # .crdownload are incomplete files, skip them
        completed = [f for f in new_files if not f.name.endswith(".crdownload")]
        if completed:
            return completed[0]
        time.sleep(1)
    raise TimeoutError("Download did not finish in time")


def main():
    driver = create_driver()
    wait = WebDriverWait(driver, 60)

    # Open the tool
    driver.get(TOOL_URL)

    # Log in if needed – the script will wait here
    input(
        "Please log in with your Omkar account in the browser and navigate to the TripAdvisor Scraper INPUT page.\n"
        "When you can see the 'Cities to Search For' field and the 'Run' / 'Download JSON' buttons, press ENTER here..."
    )

    for city in CITIES:
        print(f"\n=== Processing city: {city} ===")

        # Find the city input (update selector with Inspect if needed)
        # The selector below is a guess; you can Inspect the input on the page and copy its CSS selector if needed
        city_input = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "input[placeholder='New York']")
            )
        )
        city_input.clear()
        city_input.send_keys(city)

        # Click the Run button
        run_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'Run')]")
            )
        )
        run_button.click()
        print("Run clicked, waiting for results...")

        # After results appear, wait for the Download JSON button
        download_button = WebDriverWait(driver, 180).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'Download JSON')]")
            )
        )

        # Save existing files before starting the download
        before_files = set(DOWNLOAD_DIR.glob("*"))

        # Click Download JSON
        download_button.click()
        print("Download JSON clicked, file is downloading...")

        # Wait for download to finish and rename the file based on the city name
        downloaded_file = wait_for_download(before_files)
        new_name = DOWNLOAD_DIR / f"{city.replace(' ', '_').lower()}.json"
        downloaded_file.rename(new_name)

        print(f"JSON saved for {city}: {new_name}")


        time.sleep(3)

    print("\nAll cities finished.")
    driver.quit()


if __name__ == "__main__":
    main()
