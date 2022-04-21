from fastapi import FastAPI
app = FastAPI(title="MLOps Basics App")

@app.get("/")
async def home():
    return "<h2>This is a sample CV Project</h2>"