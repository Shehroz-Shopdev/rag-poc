from urllib.parse import urlparse  # Added urlparse
def generate_base_url(page_url):
    base_url = "https://public-api.wowcher.co.uk/v1/product-detail-page-deal/"
    
    parsed_url = urlparse(page_url)
    path_parts = parsed_url.path.strip("/").split("/")
    
    if len(path_parts) >= 5:
        region_or_deal_type = path_parts[1]  # "deal" or region like "london"
        category = path_parts[2]  # e.g., "electricals", "travel", "beauty"
        subcategory = path_parts[3]  # e.g., "laptops-macbooks", "spa"
        product_id = path_parts[4]  # Numeric product ID

        # Determine AJAX request category
        if region_or_deal_type == "deal":
            # National deals and travel deals
            if category == "travel":
                ajax_category = "travel"
                page_size = 9  # Travel uses pageSize=9
            else:
                ajax_category = "national-deal"
                page_size = 14  # Default for national deals
        else:
            # Regional deals (e.g., London)
            ajax_category = region_or_deal_type
            page_size = 14  

        base_url = f"{base_url}{ajax_category}/{category}/{subcategory}/{product_id}?page=0&pageSize={1}&offset=0"
        return base_url
    else:
        return None 

