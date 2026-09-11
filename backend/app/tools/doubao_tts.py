"""豆包语音合成与声音复刻工具。

使用豆包语音 TTS 2.0 的单向 SSE 接口。鉴权统一读取 ``VOLCANO_API_KEY``，
并将生成的音频保存到 backend/storage/audios。
"""

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "").strip()
VOLCANO_SPEECH_APP_ID = os.getenv("VOLCANO_SPEECH_APP_ID", "").strip()
VOLCANO_TTS_BASE_URL = os.getenv(
    "VOLCANO_TTS_BASE_URL", "https://openspeech.bytedance.com"
).rstrip("/")
VOLCANO_TTS_RESOURCE_ID = os.getenv("VOLCANO_TTS_RESOURCE_ID", "seed-tts-2.0").strip()
VOLCANO_TTS_SPEAKER = os.getenv(
    "VOLCANO_TTS_SPEAKER", "zh_female_vv_uranus_bigtts"
).strip()
VOLCANO_ICL_SPEAKER_ID = os.getenv("VOLCANO_ICL_SPEAKER_ID", "").strip()

STORAGE_DIR = BASE_DIR / "storage"
AUDIOS_DIR = STORAGE_DIR / "audios"
AUDIOS_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_MAP = {
    "zh": "zh-cn",
    "en": "en",
    "ja": "ja",
    "es": "es-mx",
    "id": "id",
    "pt": "pt-br",
    "ko": "ko",
}
VALID_FORMATS = {"mp3", "wav", "pcm", "ogg_opus"}


def _speech_headers(resource_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Api-Key": VOLCANO_API_KEY,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
    }


def _safe_filename(text: str, prefix: str, audio_format: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(char if char.isalnum() or char in (" ", "-", "_") else "" for char in text[:30])
    suffix = suffix.replace(" ", "_")
    filename = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
    if suffix:
        filename += f"_{suffix}"
    filename += f".{audio_format}"
    return AUDIOS_DIR / filename, f"/storage/audios/{filename}"


def _parse_sse_audio(response: requests.Response) -> tuple[bytes, str]:
    """解析豆包单向 SSE 返回的 Base64 音频数据。"""
    chunks: list[bytes] = []
    request_id = ""
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.removeprefix("data:").strip()
        if line == "[DONE]":
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        code = payload.get("code", 0)
        if code not in (0, 20000000):
            raise RuntimeError(f"豆包语音合成失败: code={code}, message={payload.get('message', '')}")
        request_id = payload.get("request_id", request_id)
        if payload.get("data"):
            chunks.append(base64.b64decode(payload["data"]))

    if not chunks:
        raise RuntimeError("豆包语音服务未返回音频数据")
    return b"".join(chunks), request_id


def _synthesize(
    *,
    text: str,
    speaker: str,
    resource_id: str,
    audio_format: str,
    speech_rate: int,
    voice_description: str = "",
    language: Optional[str] = None,
    clone_voice: bool = False,
) -> tuple[str, str]:
    """调用 TTS 2.0 SSE，并返回本地访问路径与请求 ID。"""
    if audio_format not in VALID_FORMATS:
        raise ValueError(f"audio_format 仅支持: {sorted(VALID_FORMATS)}")
    if not -50 <= speech_rate <= 100:
        raise ValueError("speech_rate 必须在 -50 到 100 之间")

    additions: dict[str, object] = {}
    if voice_description:
        additions["context_texts"] = [voice_description]
    if language:
        if language not in LANGUAGE_MAP:
            raise ValueError(f"language 仅支持 {sorted(LANGUAGE_MAP)}")
        additions["explicit_language"] = LANGUAGE_MAP[language]
    if clone_voice:
        additions["model_type"] = 4

    req_params: dict[str, object] = {
        "text": text,
        "speaker": speaker,
        "audio_params": {
            "format": audio_format,
            "sample_rate": 24000,
            "speech_rate": speech_rate,
        },
        "additions": json.dumps(additions, ensure_ascii=False),
    }
    # 复刻音色可使用 expressive 模型来获得 context_texts 的情绪控制能力。
    if clone_voice:
        req_params["model"] = "seed-tts-2.0-expressive"

    response = requests.post(
        f"{VOLCANO_TTS_BASE_URL}/api/v3/tts/unidirectional/sse",
        headers=_speech_headers(resource_id),
        json={"user": {"uid": "my-poly-studio"}, "req_params": req_params},
        stream=True,
        timeout=120,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(f"豆包语音接口请求失败: {response.text}") from error

    audio_data, request_id = _parse_sse_audio(response)
    file_path, local_path = _safe_filename(text, "doubao_tts", audio_format)
    file_path.write_bytes(audio_data)
    logger.info("✅ 豆包语音已保存: %s", file_path)
    return local_path, request_id


def _resolve_reference_audio(reference_audio: str) -> Path:
    file_path = BASE_DIR / reference_audio.lstrip("/") if reference_audio.startswith("/storage/") else Path(reference_audio)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"参考音频不存在: {file_path}")
    return file_path


def _enroll_clone_voice(reference_audio: str, speaker_id: str, language: str) -> None:
    """上传参考音频并等待 ICL 2.0 音色就绪。"""
    if not VOLCANO_SPEECH_APP_ID:
        raise ValueError("未配置 VOLCANO_SPEECH_APP_ID（声音复刻上传需要豆包语音应用 ID）")

    file_path = _resolve_reference_audio(reference_audio)
    audio_format = file_path.suffix.lower().lstrip(".")
    if audio_format not in {"wav", "mp3", "m4a", "aac"}:
        raise ValueError("参考音频仅支持 WAV、MP3、M4A 或 AAC")
    language_code = {"zh": 0, "en": 1, "ja": 2, "es": 3, "pt": 4, "id": 5}.get(language, 0)
    payload = {
        "appid": VOLCANO_SPEECH_APP_ID,
        "speaker_id": speaker_id,
        "audios": [
            {
                "audio_bytes": base64.b64encode(file_path.read_bytes()).decode("utf-8"),
                "audio_format": audio_format,
            }
        ],
        "source": 2,
        "language": language_code,
        "model_type": 4,
    }
    response = requests.post(
        f"{VOLCANO_TTS_BASE_URL}/api/v1/mega_tts/audio/upload",
        headers=_speech_headers("seed-icl-2.0"),
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status_response = requests.post(
            f"{VOLCANO_TTS_BASE_URL}/api/v1/mega_tts/status",
            headers=_speech_headers("seed-icl-2.0"),
            json={"appid": VOLCANO_SPEECH_APP_ID, "speaker_id": speaker_id},
            timeout=30,
        )
        status_response.raise_for_status()
        status = status_response.json().get("status")
        if status in (2, 4):
            return
        if status == 3:
            raise RuntimeError("豆包声音复刻训练失败")
        time.sleep(2)
    raise TimeoutError("等待豆包声音复刻就绪超时")


class DoubaoVoiceDesignInput(BaseModel):
    text: str = Field(description="要合成的文本内容")
    voice_description: str = Field(
        description="演绎指令，例如“沉稳、低沉、语速稍慢，像纪录片旁白”。"
    )
    speaker: str = Field(
        default=VOLCANO_TTS_SPEAKER,
        description="预置豆包 TTS 2.0 音色 ID；默认 Vivi 2.0。",
    )
    language: str = Field(default="zh", description="合成语言：zh、en、ja、es、id、pt、ko")
    audio_format: str = Field(default="mp3", description="输出格式：mp3、wav、pcm、ogg_opus")
    speech_rate: int = Field(default=0, description="语速，范围 -50 到 100")


@tool("doubao_voice_design", args_schema=DoubaoVoiceDesignInput)
def doubao_voice_design_tool(
    text: str,
    voice_description: str,
    speaker: str = VOLCANO_TTS_SPEAKER,
    language: str = "zh",
    audio_format: str = "mp3",
    speech_rate: int = 0,
) -> str:
    """用豆包 TTS 2.0 的预置音色按自然语言指令生成风格化语音。

    这是“音色设计”的豆包等价流程：选择预置 speaker，再用 voice_description 控制情绪和演绎；
    它不会创建一个新的自定义音色。
    """
    try:
        if not VOLCANO_API_KEY:
            return "Error: 未配置 VOLCANO_API_KEY（需要开通豆包语音 TTS 2.0 权限）"
        local_path, request_id = _synthesize(
            text=text,
            speaker=speaker,
            resource_id=VOLCANO_TTS_RESOURCE_ID,
            audio_format=audio_format,
            speech_rate=speech_rate,
            voice_description=voice_description,
            language=language,
        )
        return json.dumps(
            {
                "audio_url": local_path,
                "local_path": local_path,
                "text": text,
                "voice_description": voice_description,
                "speaker": speaker,
                "language": language,
                "provider": "volcano-doubao-tts-2.0",
                "request_id": request_id,
                "message": "豆包风格化语音生成完成",
            },
            ensure_ascii=False,
        )
    except Exception as error:
        logger.exception("❌ 豆包风格化语音生成失败")
        return f"Error: {error}"


class DoubaoVoiceCloningInput(BaseModel):
    reference_audio: str = Field(description="授权使用的参考音频路径，推荐 14 到 30 秒、单人、低噪声。")
    text: str = Field(description="要使用复刻音色合成的文本")
    consent_confirmed: bool = Field(description="确认你已获得该声音权利人的明确授权")
    speaker_id: Optional[str] = Field(
        default=None,
        description="已在豆包语音控制台创建的 S_ 开头 ICL 2.0 音色 ID；留空则使用 VOLCANO_ICL_SPEAKER_ID。",
    )
    language: str = Field(default="zh", description="参考音频语言：zh、en、ja、es、pt、id")
    voice_description: str = Field(default="", description="可选演绎指令，例如“用悲伤、克制的语气朗读”。")
    audio_format: str = Field(default="mp3", description="输出格式：mp3、wav、pcm、ogg_opus")


@tool("doubao_voice_cloning", args_schema=DoubaoVoiceCloningInput)
def doubao_voice_cloning_tool(
    reference_audio: str,
    text: str,
    consent_confirmed: bool,
    speaker_id: Optional[str] = None,
    language: str = "zh",
    voice_description: str = "",
    audio_format: str = "mp3",
) -> str:
    """通过豆包 ICL 2.0 复刻已授权声音并合成语音。

    需要在豆包语音控制台开通 ICL 2.0，并提供可用的 S_ 音色 ID 和 VOLCANO_SPEECH_APP_ID。
    """
    try:
        if not consent_confirmed:
            return "Error: 声音复刻需要声音权利人的明确授权；请确认 consent_confirmed=true。"
        if not VOLCANO_API_KEY:
            return "Error: 未配置 VOLCANO_API_KEY（需要开通豆包语音 ICL 2.0 权限）"
        selected_speaker_id = (speaker_id or VOLCANO_ICL_SPEAKER_ID).strip()
        if not selected_speaker_id.startswith("S_"):
            return "Error: 需要 S_ 开头的 ICL 2.0 speaker_id，或在 backend/.env 设置 VOLCANO_ICL_SPEAKER_ID。"
        if language not in LANGUAGE_MAP:
            return f"Error: language 仅支持 {sorted(LANGUAGE_MAP)}"

        _enroll_clone_voice(reference_audio, selected_speaker_id, language)
        local_path, request_id = _synthesize(
            text=text,
            speaker=selected_speaker_id,
            resource_id="seed-icl-2.0",
            audio_format=audio_format,
            speech_rate=0,
            voice_description=voice_description,
            language=language,
            clone_voice=True,
        )
        return json.dumps(
            {
                "audio_url": local_path,
                "local_path": local_path,
                "text": text,
                "reference_audio": reference_audio,
                "speaker_id": selected_speaker_id,
                "language": language,
                "provider": "volcano-doubao-icl-2.0",
                "request_id": request_id,
                "message": "豆包声音复刻并合成完成",
            },
            ensure_ascii=False,
        )
    except Exception as error:
        logger.exception("❌ 豆包声音复刻失败")
        return f"Error: {error}"
