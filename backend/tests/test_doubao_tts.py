import base64
import importlib
import json
import unittest
from unittest.mock import Mock, patch


MODULE_NAME = "app.tools.doubao_tts"


def _module():
    return importlib.import_module(MODULE_NAME)


class DoubaoTTSTests(unittest.TestCase):
    def test_parses_sse_audio_chunks(self):
        module = _module()
        response = Mock()
        response.iter_lines.return_value = [
            'data: {"code": 0, "data": "YQ==", "request_id": "req-1"}',
            'data: {"code": 0, "data": "Yg=="}',
            "data: [DONE]",
        ]

        audio, request_id = module._parse_sse_audio(response)

        self.assertEqual(audio, b"ab")
        self.assertEqual(request_id, "req-1")

    def test_synthesis_posts_expected_tts_v3_request(self):
        module = _module()
        response = Mock()
        response.iter_lines.return_value = [
            f'data: {{"code": 0, "data": "{base64.b64encode(b"audio").decode()}"}}',
            "data: [DONE]",
        ]
        with patch.object(module, "VOLCANO_API_KEY", "test-key"), patch.object(
            module.requests, "post", return_value=response
        ) as post, patch.object(module, "_safe_filename") as filename:
            output = Mock()
            output.write_bytes = Mock()
            filename.return_value = (output, "/storage/audios/test.mp3")
            local_path, _ = module._synthesize(
                text="你好",
                speaker="zh_female_vv_uranus_bigtts",
                resource_id="seed-tts-2.0",
                audio_format="mp3",
                speech_rate=0,
                voice_description="温柔自然",
                language="zh",
            )

        self.assertEqual(local_path, "/storage/audios/test.mp3")
        self.assertEqual(post.call_args.args[0], "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse")
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["X-Api-Key"], "test-key")
        self.assertEqual(headers["X-Api-Resource-Id"], "seed-tts-2.0")
        request_body = post.call_args.kwargs["json"]
        self.assertEqual(request_body["req_params"]["speaker"], "zh_female_vv_uranus_bigtts")
        self.assertEqual(request_body["req_params"]["audio_params"]["format"], "mp3")
        self.assertEqual(
            json.loads(request_body["req_params"]["additions"])["context_texts"],
            ["温柔自然"],
        )
        self.assertEqual(
            json.loads(request_body["req_params"]["additions"])["explicit_language"],
            "zh-cn",
        )

    def test_cloning_requires_confirmed_consent(self):
        module = _module()
        result = module.doubao_voice_cloning_tool.invoke(
            {
                "reference_audio": "/storage/audios/reference.wav",
                "text": "测试文本",
                "consent_confirmed": False,
            }
        )

        self.assertIn("明确授权", result)
