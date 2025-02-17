import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import json
import logging
import time
import re
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_driver():
    """Set up and return an undetected Chrome WebDriver with appropriate options."""
    try:
        options = uc.ChromeOptions()
        options.headless = True
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-features=site-per-process")
        options.add_argument("--window-size=1920x1080")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        )
        driver = uc.Chrome(options=options)
        return driver
    except Exception as e:
        logger.error(f"Failed to setup WebDriver: {str(e)}")
        return None

def clean_text(text):
    """Clean and format text content."""
    return ' '.join(text.strip().split()) if text else None

def is_valid_product_text(text):
    """Use regex to filter out category-like text and keep only product-relevant information."""
    if not text:
        return False
    category_pattern = re.compile(r'\b(?:deals|home|furniture|bedding|appliances|garden|fashion|beauty|fitness|discount|shopping)\b', re.IGNORECASE)
    return not category_pattern.search(text)

def scrape_product_text(url):
    """Scrape product text details and return structured data."""
    driver = setup_driver()
    if not driver:
        return None

    try:
        logger.info(f"Accessing product URL: {url}")
        driver.get(url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        product_data = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "title": None,
            "price": {"current": None, "original": None, "savings": None},
            "description": [],
            "specifications": [],
            "options": [],
        }

        # Extract title - only the actual product title, not breadcrumbs
        title_tag = soup.find("h1")
        if title_tag:
            # Clean the title to remove category paths
            title = clean_text(title_tag.get_text())
            # Only keep the last part which is typically the actual product name
            if " - " in title:
                title = title.split(" - ")[-1]
            product_data["title"] = title.strip()

        # Extract price
        price_elements = soup.find_all(
            class_=lambda x: x and any(word in str(x).lower() for word in ["price", "cost", "amount"])
        )
        for element in price_elements:
            text = clean_text(element.get_text())
            if text and "£" in text:
                if not product_data["price"]["current"]:
                    product_data["price"]["current"] = text
                elif not product_data["price"]["original"]:
                    product_data["price"]["original"] = text

        # Extract description - filtering out navigation and category lists
        description_blocks = soup.find_all(
            ["div", "section"], 
            class_=lambda x: x and any(word in str(x).lower() for word in ["description", "details", "features", "about","social-cues"])
        )
        for block in description_blocks:
            for item in block.find_all(["p", "li","span"]):
                text = clean_text(item.get_text())
                # Filter out navigation/category-like content
                if (text and len(text) > 5 and 
                    not any(word in text.lower() for word in ["shopping", "menu", "category", "navigation"]) and
                    not text.count(" ") < 2):  # Avoid single words or very short phrases
                    product_data["description"].append(text)


        meta_block = soup.find("div", class_=lambda x: x and "meta-block__info" in x)
        if meta_block:
            product_data["meta_info"] = {}
        
            bought_count = meta_block.find("p", class_=lambda x: x and "deal-main-deal__bought" in x)
            if bought_count:
                product_data["meta_info"]["bought"] = clean_text(bought_count.get_text())
        
            discount = meta_block.find("div", class_=lambda x: x and "deal-main-deal__discount" in x)
            if discount:
                product_data["meta_info"]["discount"] = clean_text(discount.get_text())
        
            previous_price = meta_block.find("div", class_=lambda x: x and "deal-main-deal__was" in x)
            if previous_price:
                product_data["meta_info"]["previous_price"] = clean_text(previous_price.get_text())
                specs_container = soup.find(
                    ["div", "section"],
                    class_=lambda x: x and any(word in str(x).lower() for word in ["specs", "highlights", "specification"])
                )
                if specs_container:
                    for item in specs_container.find_all(["li", "p", "span", "b"]):  # Include <b> to capture feature names
                        text = clean_text(item.get_text())
                        if (text and len(text) > 5 and 
                           not any(word in text.lower() for word in [
                           "shopping", "menu", "category", "navigation", 
                           "refund", "guarantee", "return policy", "warranty", "money back"
                           ]) and not text.count(" ") < 2):
                            product_data["specifications"].append(text)
        

        # Extract options - only product-specific ones like size/color
        option_elements = soup.find_all(
            ["select", "div"], 
            class_=lambda x: x and "option" in str(x).lower()
        )
        for element in option_elements:
            options = []
            for opt in element.find_all(["option", "label"]):
                text = clean_text(opt.get_text())
                if (text and len(text) > 1 and 
                    not any(word in text.lower() for word in ["select", "choose", "category"])):
                    options.append(text)
            if options:
                product_data["options"].extend(options)

        # Remove any empty lists
        product_data = {k: v for k, v in product_data.items() if v}

        filename = f"updated.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(product_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Product text data saved to {filename}")
        return product_data

    except Exception as e:
        logger.error(f"Error scraping product: {str(e)}")
        return None
    finally:
        driver.quit()


if __name__ == "__main__":
    url = "https://www.wowcher.co.uk/deal/shop/garden/garden-furniture/28303580/zero-gravity-reclining-garden-loungers"
    product_data = scrape_product_text(url)

