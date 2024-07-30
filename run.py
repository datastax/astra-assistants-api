# run.py
import os
import uvicorn

os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp"

def main():
    os.environ['DISABLE_JSON_LOGGING'] = 'true'
    uvicorn.run("impl.main:app", host="0.0.0.0", port=8000, workers=2, timeout_keep_alive=300)

if __name__ == "__main__":
    main()