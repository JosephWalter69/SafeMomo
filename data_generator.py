# data_generator.py
import pandas as pd
from faker import Faker
import random

fake = Faker('en_GB')  # British English for Ghana-like phrasing

def generate_sms_dataset(num_samples=2000):
    data = []
    for _ in range(num_samples):
        if random.random() < 0.7:  # 70% genuine
            templates = [
                f"You have received GHS {random.randint(10,1000)} from {fake.phone_number()}. Ref: {fake.uuid4()[:8]}. Balance: GHS {random.randint(500,5000)}.",
                f"MTN MoMo Alert: Cash out of GHS {random.randint(50,2000)} successful at {fake.company()}. New balance: GHS {random.randint(100,10000)}.",
                f"Your transfer of GHS {random.randint(100,5000)} to {fake.phone_number()} was successful. Ref: TXN{random.randint(100000,999999)}."
            ]
            text = random.choice(templates)
            label = 0  # 0 = genuine
        else:  # 30% fake/scam
            templates = [
                f"URGENT! Wrong transfer of GHS {random.randint(500,5000)} to your number. Call 055{random.randint(1000000,9999999)} immediately to return money!",
                f"You won GHS {random.randint(10000,100000)} prize! Click here to claim: https://mtn-claim.xyz/ref={fake.uuid4()[:6]} Confirm PIN now!",
                f"MoMo security alert: Unusual activity. Verify your account: https://secure-momo-gh.com/login?pin=required Call 024{random.randint(1000000,9999999)}!",
                f"Send back GHS {random.randint(200,3000)} sent by mistake. Do not use it! Reply with your PIN to reverse."
            ]
            text = random.choice(templates)
            label = 1  # 1 = fake/scam
        
        data.append({"text": text, "label": label})
    
    df = pd.DataFrame(data)
    df.to_csv("ghana_sms_dataset.csv", index=False)
    print(f"Generated {len(df)} samples. Saved to ghana_sms_dataset.csv")
    return df

if __name__ == "__main__":
    generate_sms_dataset(3000)  # Change number as needed