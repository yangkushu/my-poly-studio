"""文件上传路由。"""

from pathlib import Path
from uuid import uuid4
import mimetypes

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = BASE_DIR / "storage"

# 静态文件目录不会执行上传内容，但仍限制后缀，避免把任意类型文件写入存储目录。
ALLOWED_SUFFIXES = {
    "image": {".avif", ".bmp", ".gif", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".svg", ".webp"},
    "audio": {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"},
    "video": {".avi", ".mkv", ".mov", ".mp4", ".webm"},
}

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MiB
CHUNK_SIZE = 1024 * 1024


def _file_suffix(upload: UploadFile, file_type: str) -> str:
    """从原始文件名或 MIME 类型取得一个允许的后缀。"""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES[file_type]:
        guessed_suffix = mimetypes.guess_extension(upload.content_type or "") or ""
        suffix = guessed_suffix.lower()

    if suffix not in ALLOWED_SUFFIXES[file_type]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的{file_type}文件格式",
        )
    return suffix


async def _save_upload(upload: UploadFile, file_type: str) -> dict[str, str]:
    if not upload.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未收到文件")

    content_type = upload.content_type or ""
    if content_type and content_type != "application/octet-stream" and not content_type.startswith(f"{file_type}/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"文件不是有效的{file_type}类型",
        )

    suffix = _file_suffix(upload, file_type)
    directory = STORAGE_DIR / f"{file_type}s"
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"upload_{uuid4().hex}{suffix}"
    destination = directory / filename
    total_size = 0

    try:
        async with aiofiles.open(destination, "wb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="文件不能超过 100 MiB",
                    )
                await output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return {"url": f"/storage/{file_type}s/{filename}"}


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    return await _save_upload(file, "image")


@router.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    return await _save_upload(file, "audio")


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    return await _save_upload(file, "video")
