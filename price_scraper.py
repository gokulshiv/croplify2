import requests
import json
import pandas as pd
from datetime import datetime
import os

class USDALivePriceFetcher:
    def __init__(self):
        self.api_key = "9514AAE9-3FF5-33B6-8211-5C6598D09BAE"
        self.base_url = "https://quickstats.nass.usda.gov/api/api_GET"
        
        # Complete list of crops
        self.crops = [
            'Rice', 'Wheat', 'Corn', 'Barley', 'Oats', 'Sorghum', 'Millet',
            'Soybeans', 'Palm Oil', 'Canola', 'Sunflower Seeds', 'Coconut Oil',
            'Lentils', 'Chickpeas', 'Dried Beans', 'Green Peas', 'Yellow Peas',
            'Coffee', 'Tea', 'Cocoa', 'Sugar', 'Rubber', 'Black Pepper',
            'Turmeric', 'Cardamom', 'Cinnamon', 'Ginger', 'Bananas',
            'Citrus Fruits', 'Almonds', 'Cashews', 'Walnuts', 'Pistachios',
            'Hazelnuts', 'Potatoes', 'Onions', 'Tomatoes', 'Garlic', 'Apple',
            'Cotton', 'Wool', 'Orange Juice', 'Peanuts'
        ]

    def fetch_live_price(self, commodity):
        params = {
            'key': self.api_key,
            'commodity_desc': commodity.upper(),
            'statisticcat_desc': 'PRICE RECEIVED',
            'year': 2023,
            'format': 'JSON'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    latest_data = data['data'][0]
                    price_usd = float(latest_data['Value'])
                    return {
                        'price_usd': price_usd,
                        'price_inr': price_usd * 83,  # Current USD to INR rate
                        'unit': latest_data.get('unit_desc', 'USD/unit'),
                        'state': latest_data.get('state_name', 'National'),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            return None
        except Exception as e:
            print(f"Error fetching {commodity}: {str(e)}")
            return None

    def fetch_all_prices(self):
        market_prices = {}
        for crop in self.crops:
            print(f"Fetching price for {crop}...")
            price_data = self.fetch_live_price(crop)
            if price_data:
                market_prices[crop] = price_data
        return market_prices

    def save_market_data(self, prices):
        # Create data directory if not exists
        if not os.path.exists('data'):
            os.makedirs('data')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = f'data/market_prices_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(prices, f, indent=4)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(prices, orient='index')
        
        # Save as CSV
        csv_file = f'data/market_prices_{timestamp}.csv'
        df.to_csv(csv_file)
        
        # Save as Pickle
        pkl_file = f'data/market_prices_{timestamp}.pkl'
        df.to_pickle(pkl_file)
        
        # Save as latest version
        df.to_pickle('data/latest_market_prices.pkl')
        
        return {
            'json_file': json_file,
            'csv_file': csv_file,
            'pkl_file': pkl_file
        }

def main():
    print("Starting USDA Live Price Fetching Process...")
    fetcher = USDALivePriceFetcher()
    
    # Fetch all prices
    prices = fetcher.fetch_all_prices()
    
    # Save data in multiple formats
    saved_files = fetcher.save_market_data(prices)
    
    print("\nPrice fetching completed!")
    print("\nFiles saved:")
    for file_type, file_path in saved_files.items():
        print(f"{file_type}: {file_path}")
    
    print("\nSample Prices (INR):")
    for crop, data in list(prices.items())[:5]:
        print(f"{crop}: ₹{data['price_inr']:,.2f}")

if __name__ == "__main__":
    main()