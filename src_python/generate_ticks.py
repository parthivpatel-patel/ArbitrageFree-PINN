import csv
import random
import time
import os

def generate_synthetic_ticks(filename, num_ticks=1000000):
    # Ensure the data folder exists before trying to save the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Generating {num_ticks} synthetic options ticks...")
    start_time = time.time()
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'contract', 'bid', 'ask', 'volume'])
        
        # Base nanosecond timestamp
        base_ts = 1693488600000000000
        
        for _ in range(num_ticks):
            base_ts += random.randint(500, 25000)
            contract = "QQQ_260830_P_450"
            bid = round(random.uniform(5.00, 15.00), 2)
            ask = round(bid + random.uniform(0.01, 0.05), 2)
            volume = random.randint(1, 50)
            writer.writerow([base_ts, contract, bid, ask, volume])
            
    elapsed = time.time() - start_time
    print(f"Done! Created {num_ticks} ticks in {elapsed:.2f} seconds.")
    print(f"Saved to: {filename}")

if __name__ == "__main__":
    generate_synthetic_ticks("../data/raw_options_ticks.csv")