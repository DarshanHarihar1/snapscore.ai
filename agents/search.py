import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

class SearchAggregatorAgent:
    """
    Aggregates image-based search results using Google Images Lens and text queries,
    storing them per image path and query.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        options = webdriver.ChromeOptions()
        # if cfg.get('search', {}).get('headless', True):
        #     options.add_argument("--headless")
        options.add_argument("--start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        self.driver = webdriver.Chrome(options=options)
        self.timeout = cfg.get('search', {}).get('timeout', 10)
        self.wait = WebDriverWait(self.driver, self.timeout)

    def execute(self, queries_and_crops):
        """
        Perform searches for each (text_queries, crop_image_or_path) pair and store top-5 visual matches.

        Args:
            queries_and_crops: List of tuples (text_queries: List[str], crop: PIL.Image or str path)
        Returns:
            dict: Nested results {image_path: {query: [match_dict, ...], ...}, ...}
        """
        results_by_image = {}
        for idx, (queries, crop) in enumerate(queries_and_crops):
            if hasattr(crop, 'save'):
                img_path = os.path.abspath(f"temp_crop_{idx}.jpg")
                crop.save(img_path)
                remove_after = True
            else:
                img_path = os.path.abspath(crop)
                remove_after = False

            results_by_image.setdefault(img_path, {})
            for query in queries:
                results_by_image[img_path].setdefault(query, [])

                self.driver.get("https://images.google.com")
                time.sleep(2)
                lens_button = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'div[jscontroller="lpsUAf"][role="button"]'))
                )
                lens_button.click()
                time.sleep(2)
                file_input = self.wait.until(
                    EC.presence_of_element_located((By.NAME, "encoded_image"))
                )
                file_input.send_keys(img_path)
                time.sleep(2)

                try:
                    search_field = self.wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, 'textarea[placeholder="Add to your search"]'))
                    )
                except TimeoutException:
                    search_field = self.driver.find_element(By.TAG_NAME, 'textarea')
                search_field.clear()
                search_field.send_keys(query)
                search_field.send_keys(Keys.RETURN)
                time.sleep(2)

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                blocks = soup.select('div.N54PNb.BToiNc.cvP2Ce')
                for block in blocks[:5]:
                    url_tag = block.find('a', class_='LBcIee')
                    title_tag = block.select_one('span.Yt787')
                    source_tag = block.select_one('div.R8BTeb')
                    match = {
                        'title': title_tag.text.strip() if title_tag else None,
                        'url': url_tag['href'] if url_tag else None,
                        'source': source_tag.text.strip() if source_tag else None,
                    }
                    results_by_image[img_path][query].append(match)

            if remove_after:
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        self.driver.quit()

        output_path = self.cfg.get('search', {}).get('output_path')
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results_by_image, f, indent=2)

        return results_by_image