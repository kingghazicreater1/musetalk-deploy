import os
import shutil
import subprocess
from pathlib import Path

import boto3
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto",
)


class GenerateRequest(BaseModel):
    video_url: str
    audio_url: str
    job_id: str


def download_from_r2(r2_url: str, local_path: str):
    import requests
    r = requests.get(r2_url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_to_r2(local_path: str, r2_key: str) -> str:
    s3.upload_file(local_path, R2_BUCKET, r2_key)
    return f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{r2_key}"


@app.post("/generate")
async def generate(req: GenerateRequest):
    job_id = req.job_id
    work_dir = Path(f"/root/mt-results/{job_id}")
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"

    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_local = input_dir / "video.mp4"
        audio_local = input_dir / "audio.wav"
        download_from_r2(req.video_url, str(video_local))
        download_from_r2(req.audio_url, str(audio_local))

        config_dir = Path("/workspace/MuseTalk/configs/inference")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"dynamic_{job_id}.yaml"
        config_path.write_text(
            f"task_0:\n  video_path: {video_local}\n  audio_path: {audio_local}\n"
        )

        cmd = [
            "python", "-m", "scripts.inference",
            "--inference_config", str(config_path),
            "--result_dir", str(output_dir),
            "--unet_model_path", "/workspace/MuseTalk/models/musetalkV15/unet.pth",
            "--unet_config", "/workspace/MuseTalk/models/musetalkV15/musetalk.json",
            "--version", "v15",
        ]
        result = subprocess.run(
            cmd,
            cwd="/workspace/MuseTalk",
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Inference failed:\n{result.stderr[-2000:]}")

        mp4_files = list(output_dir.rglob("*.mp4"))
        if not mp4_files:
            raise RuntimeError("No MP4 output found")

        final_mp4 = mp4_files[0]
        r2_key = f"musetalk/{job_id}.mp4"
        r2_url = upload_to_r2(str(final_mp4), r2_key)

        shutil.rmtree(work_dir, ignore_errors=True)
        config_path.unlink(missing_ok=True)

        return {"r2_url": r2_url, "job_id": job_id, "status": "done"}

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)