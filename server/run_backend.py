import os
from dotenv import load_dotenv
import uvicorn

if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
