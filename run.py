# run.py
import os

import uvicorn

os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp"

def main():
    uvicorn.run("impl.main:app", host="0.0.0.0", port=8000, workers=8)

if __name__ == "__main__":
    main()