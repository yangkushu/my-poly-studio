"""fal.ai Vidu Q3 Turbo 视频生成 Tool。"""
import argparse
import json
import logging
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from app.tools.fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url
except ModuleNotFoundError:  # 支持 cd backend 后直接执行 python app/tools/<tool>.py
    from fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url

VIDU_ENDPOINTS = {
    "text": "fal-ai/vidu/q3/text-to-video/turbo",
    "image": "fal-ai/vidu/q3/image-to-video/turbo",
}


class GenerateFalViduQ3TurboVideoInput(BaseModel):
    """Vidu Q3 Turbo 的输入参数。"""

    prompt: str = Field(description="视频内容描述，最多 2000 个字符。")
    mode: Literal["text", "image"] = Field(default="text", description="text 为文生视频；image 为单图或首尾帧图生视频。")
    image_url: Optional[str] = Field(default=None, description="image 模式必填；支持公网 URL、data URI 或本地 /storage/... 路径。")
    end_image_url: Optional[str] = Field(default=None, description="image 模式可选尾帧。")
    duration: int = Field(default=5, ge=1, le=16, description="视频时长（1–16 秒）。")
    resolution: Literal["360p", "540p", "720p", "1080p"] = Field(default="720p", description="输出分辨率；有尾帧时不支持 360p。")
    aspect_ratio: Literal["16:9", "9:16", "4:3", "3:4", "1:1"] = Field(default="16:9", description="输出宽高比。")
    audio: bool = Field(default=True, description="是否生成原生音频。")
    seed: Optional[int] = Field(default=None, description="可选随机种子。")


@tool("generate_fal_vidu_q3_turbo_video", args_schema=GenerateFalViduQ3TurboVideoInput)
def generate_fal_vidu_q3_turbo_video_tool(
    prompt: str,
    mode: Literal["text", "image"] = "text",
    image_url: Optional[str] = None,
    end_image_url: Optional[str] = None,
    duration: int = 5,
    resolution: Literal["360p", "540p", "720p", "1080p"] = "720p",
    aspect_ratio: Literal["16:9", "9:16", "4:3", "3:4", "1:1"] = "16:9",
    audio: bool = True,
    seed: Optional[int] = None,
) -> str:
    """通过 fal.ai 调用 Vidu Q3 Turbo 生成视频。

    官方文档：
    - 文生：https://fal.ai/models/fal-ai/vidu/q3/text-to-video/turbo/api
    - 图生/首尾帧：https://fal.ai/models/fal-ai/vidu/q3/image-to-video/turbo/api
    - fal Python SDK：https://fal.ai/docs/api-reference/client-libraries/python/fal_client
    Q3 Turbo 仅包含文生和图生（可选尾帧）endpoint；参考图能力属于独立的 Q3 Mix，未混入本 Tool。
    价格（2026-08-05，fal 官方美元报价）：360P/540P 为 $0.035/秒，720P/1080P 为 $0.077/秒。
    按 1 USD ≈ 6.77 元估算：低分辨率 ≈ 0.24 元/秒，高分辨率 ≈ 0.52 元/秒；
    汇率会波动，实际换算值由 FAL_USD_CNY_RATE 配置决定，最终以 fal 账单为准。

    返回下载到本地 /storage/videos/ 的 video_url JSON。
    """
    if not 1 <= duration <= 16:
        return json.dumps({"error": "duration 必须为 1 至 16 秒"}, ensure_ascii=False)
    if mode == "image" and not image_url:
        return json.dumps({"error": "image 模式必须提供 image_url"}, ensure_ascii=False)
    if mode == "text" and (image_url or end_image_url):
        return json.dumps({"error": "text 模式不接受 image_url 或 end_image_url"}, ensure_ascii=False)
    if end_image_url and resolution == "360p":
        return json.dumps({"error": "提供 end_image_url 时不支持 360p"}, ensure_ascii=False)

    try:
        client = create_fal_client()
        payload = {"prompt": prompt, "duration": duration, "resolution": resolution, "aspect_ratio": aspect_ratio, "audio": audio}
        if seed is not None:
            payload["seed"] = seed
        if mode == "image":
            payload["image_url"] = resolve_input_url(image_url, client)
            if end_image_url:
                payload["end_image_url"] = resolve_input_url(end_image_url, client)
        result = generate_fal_video(VIDU_ENDPOINTS[mode], payload, prompt, "viduq3")
        price_usd = 0.035 if resolution in {"360p", "540p"} else 0.077
        return json.dumps({
            **result,
            "provider": "fal.ai", "model": "Vidu Q3 Turbo", "mode": mode,
            "endpoint": VIDU_ENDPOINTS[mode], "prompt": prompt, "resolution": resolution,
            "price_usd_per_second": price_usd, "price_yuan_per_second_estimate": round(price_usd * FAL_USD_CNY_RATE, 4),
            "usd_cny_rate_used": FAL_USD_CNY_RATE, "message": "视频已生成并保存到本地",
        }, ensure_ascii=False)
    except Exception as error:
        logging.exception("fal Vidu Q3 Turbo 生成失败")
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 FAL_KEY 会产生 fal 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 fal Vidu Q3 Turbo 视频 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image"), default="text")
    parser.add_argument("--image-url", help="image 模式的首帧")
    parser.add_argument("--end-image-url", help="image 模式可选尾帧")
    parser.add_argument("--duration", type=int, choices=range(1, 17), default=5)
    parser.add_argument("--resolution", choices=("360p", "540p", "720p", "1080p"), default="720p")
    parser.add_argument("--aspect-ratio", choices=("16:9", "9:16", "4:3", "3:4", "1:1"), default="16:9")
    parser.add_argument("--without-audio", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "image" and not args.image_url:
        parser.error("--mode image 时必须提供 --image-url")
    if args.end_image_url and args.resolution == "360p":
        parser.error("--end-image-url 时不支持 --resolution 360p")
    logging.basicConfig(level=logging.INFO)
    print(generate_fal_vidu_q3_turbo_video_tool.invoke({
        "prompt": args.prompt, "mode": args.mode, "image_url": args.image_url, "end_image_url": args.end_image_url,
        "duration": args.duration, "resolution": args.resolution, "aspect_ratio": args.aspect_ratio,
        "audio": not args.without_audio, "seed": args.seed,
    }))


if __name__ == "__main__":
    main()
