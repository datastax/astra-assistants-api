# run.py
import os
import uvicorn

os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp"

# set up logging so that file name is included
#logging.basicConfig(
#    level=logging.DEBUG,
#    format="%(asctime)s - %(levelname)s - %(message)s - %(module)s - (%(filename)s:%(lineno)d)",
#
#    datefmt="%Y-%m-%d %H:%M:%S",
#)
#logger = logging.getLogger(__name__)
def main():
    uvicorn.run("impl.main:app", host="0.0.0.0", port=8000, workers=8)

if __name__ == "__main__":
    main()