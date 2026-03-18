#!/usr/bin/env python3
"""Simple test client for TADA server."""

import sys
import requests

BASE_URL = "http://127.0.0.1:18793"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print(f"Health: {r.json()}")
    return r.ok

def test_tts(text: str, output_path: str = "test_output.wav"):
    print(f"Generating: {text[:50]}...")
    r = requests.post(f"{BASE_URL}/tts", json={"text": text})
    if r.ok:
        with open(output_path, "wb") as f:
            f.write(r.content)
        print(f"Saved to {output_path} ({len(r.content)} bytes)")
    else:
        print(f"Error: {r.status_code} - {r.text}")
    return r.ok

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is a test of the TADA text to speech system running locally."
    
    if test_health():
        test_tts(text)
    else:
        print("Server not healthy")
