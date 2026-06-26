import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.timingEndpoint:app", host="0.0.0.0", port=8082)