import requests
import pandas as pd
import time

csv_file = 'data/product_ids.csv'
df = pd.read_csv(csv_file)

print(df.head())

product_ids = df['product_id'].tolist()
shop_ids = df['shop_id'].tolist()

ratings = [1, 2, 3, 4, 5]

base_url = "https://shopee.co.id/api/v2/item/get_ratings?exclude_filter=1&filter=0&filter_size=0&flag=1&fold_filter=0&itemid={product_id}&limit=6&offset={offset}&relevant_reviews=false&request_source=2&shopid={shop_id}&tag_filter=&type={rating}0&variation_filters="

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}

for shop_id in set(shop_ids):
    reviews = [] 
    
    print(f"Start Scrape for shop_id {shop_id}")
    
    for product_id in df[df['shop_id'] == shop_id]['product_id']:
        offset = 0
        
        for rating_type in ratings:
            print(f"reviews for rating {rating_type} for shop_id {shop_id} and product_id {product_id}")
            
            while True:
                url = base_url.format(product_id=product_id, offset=offset, shop_id=shop_id, rating=rating_type)
                
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()

                if data.get('error') == 0 and data.get('data') and data['data'].get('ratings'):
                    for rating in data['data']['ratings']:
                        rating_star = rating.get('rating_star')
                        comment = rating.get('comment')
                        product_name = rating.get('product_items', [{}])[0].get('name')

                        if comment and rating_star == rating_type:
                            reviews.append({
                                "rating": rating_star,
                                "comment": comment,
                                "product_name": product_name
                            })

                    if len(data['data']['ratings']) < 6:
                        print(f"No more reviews for rating {rating_type} for shop_id {shop_id} and product_id {product_id}.")
                        break
                else:
                    print(f"No ratings data found for rating {rating_type} for shop_id {shop_id} and product_id {product_id}.")
                    break

                offset += 6
                time.sleep(1) 

    if reviews:
        filename = f"data/shopee_scincare_reviews_{shop_id}.csv"
        df_reviews = pd.DataFrame(reviews)
        df_reviews.to_csv(filename, index=False)
        print(f"Reviews for shop_id {shop_id} saved as {filename}.")
    else:
        print(f"No reviews found for shop_id {shop_id}.")
