from urllib.parse import urlparse

def generate_base_url(page_url):
    base_url = "https://public-api.wowcher.co.uk/v1/product-detail-page-deal/"
    
    parsed_url = urlparse(page_url)
    path_parts = parsed_url.path.strip("/").split("/")
    
    if len(path_parts) >= 5:
        region_or_deal_type = path_parts[1]  
        category = path_parts[2]  
        subcategory = path_parts[3]  
        product_id = path_parts[4]  

        if region_or_deal_type == "deal":
            if category == "travel":
                ajax_category = "travel"
            else:
                ajax_category = "national-deal"
        else:
            ajax_category = region_or_deal_type

        base_url = f"{base_url}{ajax_category}/{category}/{subcategory}/{product_id}?page=0&pageSize={1}&offset=0"
        return base_url
    else:
        return None 