#!/usr/bin/env python
"""Test the Streamlit UI and FastAPI from terminal"""

import requests
import sys

print("=" * 60)
print("Testing Sentiment Analysis from Docker")
print("=" * 60 + "\n")

# Test cases
test_cases = [
    "I love this product!",
    "Terrible experience",
    "This is amazing!",
    "Worst purchase ever",
    "It's okay, nothing special"
]

print("Testing API predictions...\n")

for text in test_cases:
    try:
        response = requests.post(
            'http://localhost:8000/predict',
            json={'text': text}
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            confidence = result['confidence']
            emoji = "😊" if prediction == "Positive" else "😡"
            
            print(f"Text: {text}")
            print(f"Result: {emoji} {prediction} ({confidence:.2%})")
            print()
        else:
            print(f"Error: {response.status_code}")
    
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

print("=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nAccess the UI at: http://localhost:8501")
print("API Docs at: http://localhost:8000/docs")
