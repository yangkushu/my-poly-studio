"""fal.ai Kling O3 Standard 视频生成 Tool。"""
import argparse
import json
import logging
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from app.tools.fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url
except ModuleNotFoundError:  # 支持 cd backend 后直接执行 python app/tools/<tool>.py
    from fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url

KLING_ENDPOINTS = {
    "text": "fal-ai/kling-video/o3/standard/text-to-video",
    "image": "fal-ai/kling-video/o3/standard/image-to-video",
    "reference": "fal-ai/kling-video/o3/standard/reference-to-video",
}


class GenerateFalKlingO3StandardVideoInput(BaseModel):
    """Kling O3 Standard 视频生成工具的输入参数。"""

    prompt: str = Field(
        description="必填。描述画面、主体动作、镜头和风格。reference 模式中，使用 @Image1、@Image2 等引用 image_urls 中对应的参考图。"
    )
    mode: Literal["text", "image", "reference"] = Field(
        default="text",
        description="生成模式：text=仅凭提示词生成，不能传图片；image=使用一张首帧生成，可选尾帧；reference=使用 1–4 张参考图保持人物或物体的外观/风格一致。",
    )
    image_urls: list[str] = Field(
        default_factory=list,
        description="图片列表。text 模式必须为空；image 模式必须且只能传 1 张首帧；reference 模式必须传 1–4 张参考图（不是首帧）。每项可为公网 URL、data URI 或本项目的 /storage/... 图片路径。",
    )
    end_image_url: Optional[str] = Field(
        default=None,
        description="可选尾帧图片，仅 image 或 reference 模式可用；视频将以该图作为最后一帧。可为公网 URL、data URI 或本项目的 /storage/... 图片路径。",
    )
    duration: Literal["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"] = Field(
        default="5", description="视频时长（秒），可选 3–15，默认 5。"
    )
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = Field(
        default="16:9",
        description="text 和 reference 模式的输出画幅比：16:9、9:16 或 1:1，默认 16:9。image 模式的 fal 端点不支持此参数，本工具会忽略它。",
    )
    generate_audio: bool = Field(
        default=False, description="是否请求模型生成视频的原生音频，默认否。"
    )


@tool("generate_fal_kling_o3_standard_video", args_schema=GenerateFalKlingO3StandardVideoInput)
def generate_fal_kling_o3_standard_video_tool(
    prompt: str,
    mode: Literal["text", "image", "reference"] = "text",
    image_urls: Optional[list[str]] = None,
    end_image_url: Optional[str] = None,
    duration: Literal["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"] = "5",
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9",
    generate_audio: bool = False,
) -> str:
    """
    使用 fal.ai 的 Kling O3 Standard 生成视频。

    按需求选择一种模式：
    - text：纯文生视频；只提供 prompt，不传 image_urls 或 end_image_url。
    - image：首帧图生视频；image_urls 必须仅含一张首帧，end_image_url 可选，用于指定尾帧。
    - reference：参考图生视频；image_urls 必须含 1–4 张参考图。若要在提示词中指定某张图的角色、物体或风格，分别写 @Image1、@Image2 等。end_image_url 可选。

    支持 3–15 秒视频，并可请求生成原生音频。工具完成后返回包含本地 video_url 的 JSON 结果。
    """
    image_urls = image_urls or []
    if mode == "text" and (image_urls or end_image_url):
        logger.warning("Kling O3 视频请求被拒绝: text 模式包含图片参数")
        return json.dumps({"error": "text 模式不接受 image_urls 或 end_image_url"}, ensure_ascii=False)
    if mode == "image" and len(image_urls) != 1:
        logger.warning("Kling O3 视频请求被拒绝: image 模式图片数量=%s", len(image_urls))
        return json.dumps({"error": "image 模式需要恰好 1 张首帧图片"}, ensure_ascii=False)
    if mode == "reference" and not 1 <= len(image_urls) <= 4:
        logger.warning("Kling O3 视频请求被拒绝: reference 模式图片数量=%s", len(image_urls))
        return json.dumps({"error": "reference 模式需要 1 至 4 张参考图"}, ensure_ascii=False)

    try:
        logger.info(
            "开始 Kling O3 视频生成: mode=%s duration=%ss aspect_ratio=%s generate_audio=%s image_count=%s has_end_frame=%s",
            mode, duration, aspect_ratio if mode != "image" else "n/a", generate_audio, len(image_urls), bool(end_image_url),
        )
        client = create_fal_client()
        payload = {"prompt": prompt, "duration": duration, "generate_audio": generate_audio}
        # fal 的 Kling O3 Standard image-to-video 端点没有 aspect_ratio 参数。
        if mode != "image":
            payload["aspect_ratio"] = aspect_ratio
        if mode == "image":
            payload["image_url"] = resolve_input_url(image_urls[0], client)
        elif mode == "reference":
            payload["image_urls"] = [resolve_input_url(image_url, client) for image_url in image_urls]
        if end_image_url:
            payload["end_image_url"] = resolve_input_url(end_image_url, client)
        result = generate_fal_video(KLING_ENDPOINTS[mode], payload, prompt, "klingo3")
        logger.info("Kling O3 视频生成完成: task_id=%s local_path=%s", result.get("task_id"), result.get("local_path"))
        price_usd = 0.112 if generate_audio else 0.084
        return json.dumps({
            **result,
            "provider": "fal.ai", "model": "Kling O3 Standard", "mode": mode,
            "endpoint": KLING_ENDPOINTS[mode], "prompt": prompt,
            "price_usd_per_second": price_usd, "price_yuan_per_second_estimate": round(price_usd * FAL_USD_CNY_RATE, 4),
            "usd_cny_rate_used": FAL_USD_CNY_RATE, "message": "视频已生成并保存到本地",
        }, ensure_ascii=False)
    except Exception as error:
        logger.exception("Kling O3 视频生成失败: mode=%s", mode)
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 FAL_KEY 会产生 fal 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 fal Kling O3 Standard 视频 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image", "reference"), default="text")
    parser.add_argument("--image-url", dest="image_urls", action="append", help="首帧或参考图；可重复传入 reference 图片")
    parser.add_argument("--end-image-url", help="image/reference 模式可选尾帧")
    parser.add_argument("--duration", choices=tuple(str(value) for value in range(3, 16)), default="5")
    parser.add_argument("--aspect-ratio", choices=("16:9", "9:16", "1:1"), default="16:9")
    parser.add_argument("--generate-audio", action="store_true")
    args = parser.parse_args()
    image_urls = args.image_urls or []
    if args.mode == "image" and len(image_urls) != 1:
        parser.error("--mode image 时必须恰好提供一个 --image-url")
    if args.mode == "reference" and not 1 <= len(image_urls) <= 4:
        parser.error("--mode reference 时必须提供 1 至 4 个 --image-url")
    if args.mode == "text" and (image_urls or args.end_image_url):
        parser.error("--mode text 不接受图片参数")
    logging.basicConfig(level=logging.INFO)
    print(generate_fal_kling_o3_standard_video_tool.invoke({
        "prompt": args.prompt, "mode": args.mode, "image_urls": image_urls, "end_image_url": args.end_image_url,
        "duration": args.duration, "aspect_ratio": args.aspect_ratio, "generate_audio": args.generate_audio,
    }))


if __name__ == "__main__":
    main()
