import os
import shutil
import subprocess
from pathlib import Path

import boto3
import runpod

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


def download_from_r2(r2_url: str, local_path: str):
    import requests
    r = requests.get(r2_url, stream=True, timeout=120)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_to_r2(local_path: str, r2_key: str) -> str:
    s3.upload_file(local_path, R2_BUCKET, r2_key)
    return f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{r2_key}"


def _ensure_models():
    target = Path("/workspace/MuseTalk/models")
    need = Path("musetalkV15") / "unet.pth"
    if (target / need).exists():
        return
    for base in ["/runpod-volume/musetalk-models", "/runpod-volume/models",
                 "/runpod-volume/MuseTalk/models", "/runpod-volume"]:
        if (Path(base) / need).exists():
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            target.symlink_to(base)
            return
    raise RuntimeError("MuseTalk weights nahi mile (musetalkV15/unet.pth)")


def _debug_listing():
    out = {}
    for p in ["/runpod-volume", "/workspace", "/workspace/MuseTalk/models", "/workspace/musetalk-models"]:
        pp = Path(p)
        out[p] = sorted(x.name for x in pp.iterdir())[:50] if pp.exists() else "MISSING"
    pkg = Path("/workspace/MuseTalk/musetalk")
    out["musetalk_pkg"] = sorted(x.name for x in pkg.iterdir()) if pkg.exists() else "MISSING"
    return out


def handler(job):
    job_input = job["input"]
    if job_input.get("debug"):
        return _debug_listing()

    _ensure_models()

    video_url = job_input["video_url"]
    audio_url = job_input["audio_url"]
    job_id = job_input["job_id"]

    work_dir = Path(f"/root/mt-results/{job_id}")
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    config_path = Path(f"/workspace/MuseTalk/configs/inference/dynamic_{job_id}.yaml")

    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_local = input_dir / "video.mp4"
        audio_local = input_dir / "audio.wav"
        download_from_r2(video_url, str(video_local))
        download_from_r2(audio_url, str(audio_local))

        config_path.parent.mkdir(parents=True, exist_ok=True)
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
            cmd, cwd="/workspace/MuseTalk",
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Inference failed:\n{result.stderr[-2000:]}")

        mp4_files = list(output_dir.rglob("*.mp4"))
        if not mp4_files:
            raise RuntimeError("No MP4 output found")

        r2_key = f"musetalk/{job_id}.mp4"
        r2_url = upload_to_r2(str(mp4_files[0]), r2_key)
        return {"r2_url": r2_url, "job_id": job_id, "status": "done"}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        config_path.unlink(missing_ok=True)


runpod.serverless.start({"handler": handler})
