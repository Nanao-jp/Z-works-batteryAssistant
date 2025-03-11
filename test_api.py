"""
Ollama API Test Script - Fixed encoding issues
"""

import os
import requests
import sys
import json
import re
import urllib.parse
import urllib3
from dotenv import load_dotenv

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load settings from .env file
load_dotenv()

# Print debug info
print(f"Python version: {sys.version}")
print(f"Default encoding: {sys.getdefaultencoding()}")

# API settings - ensuring ASCII characters only
api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
api_key = os.getenv("OLLAMA_API_KEY", "")
model_name = os.getenv("MODEL_NAME", "llama3")

# Check for non-ASCII characters in URL
def check_url(url):
    print(f"Checking URL: {url}")
    print(f"URL contains non-ASCII characters: {not all(ord(c) < 128 for c in url)}")
    if not all(ord(c) < 128 for c in url):
        print("URL encoding needed. Applying URL encoding...")
        # Find the problematic characters
        for i, c in enumerate(url):
            if ord(c) >= 128:
                print(f"Non-ASCII char at position {i}: {c} (code: {ord(c)})")
        
        # Try to encode the URL properly
        parsed = urllib.parse.urlparse(url)
        safe_url = parsed.scheme + "://" + parsed.netloc + urllib.parse.quote(parsed.path)
        if parsed.query:
            safe_url += "?" + parsed.query
        print(f"Encoded URL: {safe_url}")
        return safe_url
    return url

# Test with a mock API that accepts any request
def test_with_mock():
    """Test with a mock API that always returns success"""
    print("Testing with httpbin.org (mock API)...")
    
    data = {
        "model": model_name,
        "prompt": "Please explain briefly about AI development.",
        "stream": False
    }
    
    try:
        # Send to a test endpoint that echoes the request
        response = requests.post(
            "https://httpbin.org/post",
            json=data
        )
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            response_data = response.json()
            print(f"Request sent to: {response_data['url']}")
            print("Mock API request successful!")
            return True
        else:
            print(f"Mock API request failed: {response.text}")
            return False
    except Exception as e:
        print(f"Exception during mock test: {str(e)}")
        return False

def test_url_only():
    """Just test the ability to connect to the API URL without sending data"""
    try:
        url = check_url(f"{api_base}/api/tags")
        print(f"Testing connection to: {url}")
        
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Exception testing URL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_alternative():
    """Alternative approach using subprocess to curl directly"""
    print("Testing API using an alternative approach...")
    
    try:
        # Use socket library directly to avoid header encoding issues
        import socket
        import ssl
        
        # Parse URL to get host and path
        parsed_url = urllib.parse.urlparse(api_base)
        host = parsed_url.netloc
        
        # Create socket connection
        is_https = parsed_url.scheme == 'https'
        port = 443 if is_https else 80
        
        # Remove any port if it's in the host
        if ':' in host:
            host, port_str = host.split(':')
            port = int(port_str)
            
        print(f"Connecting to {host}:{port}...")
        
        # Create the socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Wrap with SSL if needed
        if is_https:
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=host)
        
        # Connect
        sock.connect((host, port))
        
        # Prepare JSON data
        data = {
            "model": model_name,
            "prompt": "Tell me about AI",
            "stream": False
        }
        data_json = json.dumps(data)
        
        # Prepare the HTTP POST request
        headers = [
            f"POST /api/generate HTTP/1.1",
            f"Host: {host}",
            f"Content-Type: application/json",
            f"Content-Length: {len(data_json)}",
            "Connection: close",
            ""
        ]
        
        if api_key:
            headers.insert(3, f"Authorization: Bearer {api_key}")
        
        # Make the request
        request = "\r\n".join(headers) + "\r\n" + data_json
        
        print("Sending request:")
        print(request)
        
        # Send the request
        sock.sendall(request.encode('ascii'))
        
        # Receive the response
        response = b""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data
            
        # Close the connection
        sock.close()
        
        # Parse the response
        response_text = response.decode('utf-8')
        print("Response received:")
        print(response_text[:500])  # Print first 500 chars
        
        return True
    
    except Exception as e:
        print(f"Exception during alternative test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting API tests...")
    
    # First test with mock API to verify request formatting works
    if test_with_mock():
        print("\nNow testing API URL connectivity...")
        if test_url_only():
            print("\nURL connection successful!")
            
            print("\nNow testing alternative approach...")
            if test_ollama_alternative():
                print("\nAlternative API test completed!")
            else:
                print("\nAlternative API test failed.")
        else:
            print("\nURL connection failed. Please check your API endpoint.")
    else:
        print("\nMock test failed, not attempting Ollama API tests.") 