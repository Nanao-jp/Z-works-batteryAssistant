import os
import uvicorn
from api_server import app

if __name__ == "__main__":
    # Get port from environment variable or use 8000 as default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application with uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, log_level="info") 