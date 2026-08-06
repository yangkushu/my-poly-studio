"""fal.ai Vidu Q3 Turbo 视频生成 Tool。"""
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

VIDU_ENDPOINTS = {
    "text": "fal-ai/vidu/q3/text-to-video/turbo",
    "image": "fal-ai/vidu/q3/image-to-video/turbo",
}


class GenerateFalViduQ3TurboVideoInput(BaseModel):
    """Vidu Q3 Turbo 视频生成工具的输入参数。"""

    prompt: str = Field(description="必填。视频内容描述，最多 2000 个字符；image 模式也用它描述图片应如何运动、镜头如何变化。")
    mode: Literal["text", "image"] = Field(
        default="text",
        description="生成模式：text=仅凭提示词生成，不能传图片；image=使用一张首帧生成，可选尾帧以生成首尾帧过渡视频。",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="首帧图片。仅 image 模式必填；可为公网 URL、data URI 或本项目的 /storage/... 图片路径。",
    )
    end_image_url: Optional[str] = Field(
        default=None,
        description="可选尾帧图片，仅 image 模式可用；提供后生成从 image_url 到此图的过渡视频。可为公网 URL、data URI 或本项目的 /storage/... 图片路径。",
    )
    duration: int = Field(default=5, ge=1, le=16, description="视频时长（秒），范围 1–16，默认 5。")
    resolution: Literal["360p", "540p", "720p", "1080p"] = Field(
        default="720p", description="输出分辨率，默认 720p；提供 end_image_url 时不能选 360p。"
    )
    aspect_ratio: Literal["16:9", "9:16", "4:3", "3:4", "1:1"] = Field(
        default="16:9",
        description="text 模式的输出画幅比：16:9、9:16、4:3、3:4 或 1:1，默认 16:9。image 端点不支持此参数，本工具会忽略它。",
    )
    audio: bool = Field(default=True, description="是否启用音画同步生成；启用后视频可包含对白和音效，默认启用。")
    seed: Optional[int] = Field(default=None, description="可选随机种子，用于提高重复生成时的可复现性；省略则由服务随机选择。")


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
    """
    使用 fal.ai 的 Vidu Q3 Turbo 生成视频。

    - text：纯文生视频；只提供 prompt，不传 image_url 或 end_image_url。
    - image：首帧图生视频；必须提供 image_url。可选 end_image_url，用于生成首帧到尾帧的过渡视频。

    本工具不支持参考图模式。视频可选 1–16 秒，并可生成包含对白和音效的原生音频。完成后返回包含本地 video_url 的 JSON 结果。
    """
    if not 1 <= duration <= 16:
        logger.warning("Vidu Q3 视频请求被拒绝: 非法时长 duration=%s", duration)
        return json.dumps({"error": "duration 必须为 1 至 16 秒"}, ensure_ascii=False)
    if mode == "image" and not image_url:
        logger.warning("Vidu Q3 视频请求被拒绝: image 模式缺少首帧")
        return json.dumps({"error": "image 模式必须提供 image_url"}, ensure_ascii=False)
    if mode == "text" and (image_url or end_image_url):
        logger.warning("Vidu Q3 视频请求被拒绝: text 模式包含图片参数")
        return json.dumps({"error": "text 模式不接受 image_url 或 end_image_url"}, ensure_ascii=False)
    if end_image_url and resolution == "360p":
        logger.warning("Vidu Q3 视频请求被拒绝: 尾帧不支持 360p")
        return json.dumps({"error": "提供 end_image_url 时不支持 360p"}, ensure_ascii=False)

    try:
        logger.info(
            "开始 Vidu Q3 视频生成: mode=%s duration=%ss resolution=%s aspect_ratio=%s audio=%s has_end_frame=%s has_seed=%s",
            mode, duration, resolution, aspect_ratio if mode == "text" else "n/a", audio, bool(end_image_url), seed is not None,
        )
        client = create_fal_client()
        payload = {"prompt": prompt, "duration": duration, "resolution": resolution, "audio": audio}
        # fal 的 Vidu Q3 image-to-video 端点没有 aspect_ratio 参数。
        if mode == "text":
            payload["aspect_ratio"] = aspect_ratio
        if seed is not None:
            payload["seed"] = seed
        if mode == "image":
            payload["image_url"] = resolve_input_url(image_url, client)
            if end_image_url:
                payload["end_image_url"] = resolve_input_url(end_image_url, client)
        result = generate_fal_video(VIDU_ENDPOINTS[mode], payload, prompt, "viduq3")
        logger.info("Vidu Q3 视频生成完成: task_id=%s local_path=%s", result.get("task_id"), result.get("local_path"))
        price_usd = 0.035 if resolution in {"360p", "540p"} else 0.077
        return json.dumps({
            **result,
            "provider": "fal.ai", "model": "Vidu Q3 Turbo", "mode": mode,
            "endpoint": VIDU_ENDPOINTS[mode], "prompt": prompt, "resolution": resolution,
            "price_usd_per_second": price_usd, "price_yuan_per_second_estimate": round(price_usd * FAL_USD_CNY_RATE, 4),
            "usd_cny_rate_used": FAL_USD_CNY_RATE, "message": "视频已生成并保存到本地",
        }, ensure_ascii=False)
    except Exception as error:
        logger.exception("Vidu Q3 视频生成失败: mode=%s", mode)
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
