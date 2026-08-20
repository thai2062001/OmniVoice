#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import re
import tempfile
import zipfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


def map_to_valid_instruct(emotion_str: str, guidance_str: str):
    """
    Maps freeform emotion/guidance keywords from script to valid OmniVoice attributes.
    Ensures only attributes strictly supported by OmniVoice are passed.
    """
    from omnivoice.utils.voice_design import _INSTRUCT_ALL_VALID
    
    combined = f"{emotion_str} {guidance_str}".lower()
    valid_tags = []
    
    # Style
    if any(k in combined for k in ["whisper", "thì thầm", "耳语"]):
        valid_tags.append("whisper")
        
    # Pitch / Energy
    if any(k in combined for k in ["high energy", "energetic", "excited", "surprised", "high pitch", "cao trào", "vui vẻ", "ngạc nhiên", "sôi nổi"]):
        valid_tags.append("high pitch")
    elif any(k in combined for k in ["very high pitch", "hét", "screaming"]):
        valid_tags.append("very high pitch")
    elif any(k in combined for k in ["low pitch", "serious", "calm", "trầm", "nghiêm túc"]):
        valid_tags.append("low pitch")
    elif any(k in combined for k in ["very low pitch", "deep voice"]):
        valid_tags.append("very low pitch")
    elif any(k in combined for k in ["informative", "steady pace", "steady", "moderate pitch", "bình thường"]):
        valid_tags.append("moderate pitch")
        
    # Age
    if any(k in combined for k in ["child", "kid", "trẻ em", "em bé"]):
        valid_tags.append("child")
    elif any(k in combined for k in ["teenager", "teen", "thiếu niên"]):
        valid_tags.append("teenager")
    elif any(k in combined for k in ["elderly", "old", "người già", "ông lão", "bà lão"]):
        valid_tags.append("elderly")
    elif any(k in combined for k in ["young adult", "young", "trẻ"]):
        valid_tags.append("young adult")
        
    # Gender
    if any(k in combined for k in ["female", "woman", "girl", "nữ", "gái"]):
        valid_tags.append("female")
    elif any(k in combined for k in ["male", "man", "boy", "nam", "trai"]):
        valid_tags.append("male")
        
    # Accents
    if "japanese" in combined or "tiếng nhật" in combined:
        valid_tags.append("japanese accent")
    elif "british" in combined:
        valid_tags.append("british accent")
    elif "american" in combined:
        valid_tags.append("american accent")
    elif "chinese" in combined:
        valid_tags.append("chinese accent")

    filtered = []
    for t in valid_tags:
        if t in _INSTRUCT_ALL_VALID and t not in filtered:
            filtered.append(t)
    return ", ".join(filtered) if filtered else None


def parse_script(script_text: str):
    """
    Parses a script of segments with timelines, text, emotions and instructions.
    """
    segments = []
    lines = script_text.split('\n')
    current_seg = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        time_match = re.match(r'\[#(\d+)\]\s*THỜI GIAN:\s*([\d\.]+)\s*->\s*([\d\.]+)', line, re.IGNORECASE)
        if time_match:
            if current_seg:
                segments.append(current_seg)
            seg_id = int(time_match.group(1))
            start = float(time_match.group(2))
            end = float(time_match.group(3))
            current_seg = {
                "id": seg_id,
                "duration": round(end - start, 2),
                "text": "",
                "emotion": "",
                "guidance": "",
            }
            continue
            
        if current_seg is None:
            continue
            
        text_match = re.match(r'VĂN BẢN\s*(?:\([^)]+\))?:\s*(.+)', line, re.IGNORECASE)
        if text_match:
            current_seg["text"] = text_match.group(1).strip()
            continue
            
        emotion_match = re.match(r'CẢM XÚC:\s*(.+)', line, re.IGNORECASE)
        if emotion_match:
            current_seg["emotion"] = emotion_match.group(1).strip()
            continue
            
        guidance_match = re.match(r'HƯỚNG DẪN AI:\s*(.+)', line, re.IGNORECASE)
        if guidance_match:
            current_seg["guidance"] = guidance_match.group(1).strip()
            continue
            
    if current_seg:
        segments.append(current_seg)
        
    formatted_segs = []
    for seg in segments:
        instruct_parts = []
        if seg["emotion"]:
            instruct_parts.append(seg["emotion"])
        if seg["guidance"]:
            instruct_parts.append(seg["guidance"])
            
        valid_inst = map_to_valid_instruct(seg["emotion"], seg["guidance"])
            
        formatted_segs.append({
            "id": seg["id"],
            "duration": seg["duration"],
            "text": seg["text"],
            "raw_instruct": ", ".join(instruct_parts),
            "valid_instruct": valid_inst
        })
    return formatted_segs


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        voice_clone_prompt=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if voice_clone_prompt is not None:
                kw["voice_clone_prompt"] = voice_clone_prompt
            else:
                if not ref_audio:
                    return None, "Please upload a reference audio."
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed. Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional) / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Batch Voice Clone
            # ==============================================================
            with gr.TabItem("Batch Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bvc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        
                        gr.Markdown("### Shared Reference Voice / 共享参考声音")
                        with gr.Group():
                            bvc_ref_audio = gr.Audio(
                                label="Shared Reference Audio / 共享参考音频",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            bvc_ref_text = gr.Textbox(
                                label="Shared Reference Text (optional) / 共享参考文本（可选）",
                                placeholder="Transcript of the reference audio. Leave empty to auto-transcribe.",
                            )
                        
                        gr.Markdown("### Texts to Synthesize / 待合成文本 (Up to 5)")
                        bvc_text1 = gr.Textbox(
                            label="Text 1 / 文本 1",
                            placeholder="Enter the text for Voice 1...",
                        )
                        bvc_text2 = gr.Textbox(
                            label="Text 2 / 文本 2",
                            placeholder="Enter the text for Voice 2...",
                        )
                        bvc_text3 = gr.Textbox(
                            label="Text 3 / 文本 3",
                            placeholder="Enter the text for Voice 3...",
                        )
                        bvc_text4 = gr.Textbox(
                            label="Text 4 / 文本 4",
                            placeholder="Enter the text for Voice 4...",
                        )
                        bvc_text5 = gr.Textbox(
                            label="Text 5 / 文本 5",
                            placeholder="Enter the text for Voice 5...",
                        )

                        with gr.Accordion("Instruct (optional)", open=False):
                            bvc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            bvc_ns,
                            bvc_gs,
                            bvc_dn,
                            bvc_sp,
                            bvc_du,
                            bvc_pp,
                            bvc_po,
                        ) = _gen_settings()
                        bvc_btn = gr.Button("Generate All / Đồng loạt sinh giọng", variant="primary")
                    with gr.Column(scale=1):
                        bvc_audio1 = gr.Audio(
                            label="Output Voice 1 / 合成结果 1",
                            type="numpy",
                        )
                        bvc_audio2 = gr.Audio(
                            label="Output Voice 2 / 合成结果 2",
                            type="numpy",
                        )
                        bvc_audio3 = gr.Audio(
                            label="Output Voice 3 / 合成结果 3",
                            type="numpy",
                        )
                        bvc_audio4 = gr.Audio(
                            label="Output Voice 4 / 合成结果 4",
                            type="numpy",
                        )
                        bvc_audio5 = gr.Audio(
                            label="Output Voice 5 / 合成结果 5",
                            type="numpy",
                        )
                        bvc_status = gr.Textbox(label="Status / 状态", lines=5)

                def _batch_clone_fn(
                    lang, 
                    ref_audio,
                    ref_text,
                    text1,
                    text2,
                    text3,
                    text4,
                    text5,
                    instruct, ns, gs, dn, sp, du, pp, po
                ):
                    results = []
                    statuses = []
                    
                    if not ref_audio:
                        return None, None, None, None, None, "Error: Please upload a shared reference audio."
                    
                    try:
                        # Pre-generate clone prompt
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=ref_audio,
                            ref_text=ref_text or None,
                        )
                    except Exception as e:
                        return None, None, None, None, None, f"Error encoding reference audio: {e}"
                    
                    texts = [text1, text2, text3, text4, text5]
                    
                    for i, t in enumerate(texts, 1):
                        if t and t.strip():
                            try:
                                res, stat = _gen(
                                    t.strip(),
                                    lang,
                                    ref_audio,
                                    instruct,
                                    ns,
                                    gs,
                                    dn,
                                    sp,
                                    du,
                                    pp,
                                    po,
                                    mode="clone",
                                    ref_text=ref_text or None,
                                    voice_clone_prompt=prompt,
                                )
                                results.append(res)
                                statuses.append(f"Voice {i}: {stat}")
                            except Exception as e:
                                results.append(None)
                                statuses.append(f"Voice {i}: Error: {e}")
                        else:
                            results.append(None)
                            statuses.append(f"Voice {i}: Skipped (empty text)")
                            
                    return results[0], results[1], results[2], results[3], results[4], "\n".join(statuses)

                bvc_btn.click(
                    _batch_clone_fn,
                    inputs=[
                        bvc_lang,
                        bvc_ref_audio,
                        bvc_ref_text,
                        bvc_text1,
                        bvc_text2,
                        bvc_text3,
                        bvc_text4,
                        bvc_text5,
                        bvc_instruct,
                        bvc_ns,
                        bvc_gs,
                        bvc_dn,
                        bvc_sp,
                        bvc_du,
                        bvc_pp,
                        bvc_po,
                    ],
                )

            # ==============================================================
            # Script Clone
            # ==============================================================
            with gr.TabItem("Script Clone / Sinh giọng theo Kịch bản"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        
                        gr.Markdown("### 1. Upload Shared Reference Voice / Tải lên giọng mẫu dùng chung")
                        with gr.Group():
                            sc_ref_audio = gr.Audio(
                                label="Shared Reference Audio (Optional for Voice Cloning) / Giọng nói mẫu dùng chung (Tùy chọn)",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            sc_ref_text = gr.Textbox(
                                label="Shared Reference Text (optional) / Văn bản giọng mẫu (Tùy chọn)",
                                placeholder="Leave empty to auto-transcribe.",
                            )
                        
                        gr.Markdown("### 2. Paste Script / Dán Kịch bản timeline")
                        sc_script = gr.Textbox(
                            label="Script / Kịch bản",
                            lines=12,
                            placeholder="[#1] THỜI GIAN: 0.0 -> 5.0\nVĂN BẢN (JP): 日本のコンビニ...\nCẢM XÚC: Energetic\nHƯỚNG DẪN AI: High energy intro\n------------------------------------------",
                            value="""BẢN TỔNG HỢP TIMELINE & HIỆU ỨNG ÂM THANH
==========================================
Thời lượng tổng: 30.0
Số lượng phân đoạn: 5

[#1] THỜI GIAN: 0.0 -> 5.0
VĂN BẢN (JP): 日本のコンビニ、実は海外から見ると「魔法の場所」らしいよ。外国人がガチで衝撃を受けた3つのこと、紹介するね！
HIỆU ỨNG (SFX): Paper tearing sound
CẢM XÚC: Energetic
HƯỚNG DẪN AI: High energy intro
------------------------------------------
[#2] THỜI GIAN: 5.0 -> 11.5
VĂN BẢN (JP): 第3位、レジだけで生活が完結。公共料金の支払いから宅配の受け取りまで、何でもできちゃう。
HIỆU ỨNG (SFX): Register 'ding'
CẢM XÚC: Informative
HƯỚNG DẪN AI: Steady pace
------------------------------------------
[#3] THỜI GIAN: 12.0 -> 19.5
VĂN BẢN (JP): 第2位、深夜でも揚げたてフード。店員さんが作るチキンやおでん、クオリティ高すぎ！
HIỆU ỨNG (SFX): Sizzling sound
CẢM XÚC: Excited
HƯỚNG DẪN AI: Emphasize 'quality'
------------------------------------------
[#4] THỜI GIAN: 20.0 -> 26.5
VĂN BẢN (JP): そして第1位は、どこでも無料のトイレ。しかもピカピカ！これ、海外じゃマジで奇跡んだって。
HIỆU ỨNG (SFX): Sparkle chime
CẢM XÚC: Surprised
HƯỚNG DẪN AI: Slow down for impact on 'miracle'
------------------------------------------
[#5] THỜI GIAN: 27.0 -> 30.0
VĂN BẢN (JP): 旅行者には最高の味方だよね。日本のコンビニ、すごすぎない？
HIỆU ỨNG (SFX): Pop
CẢM XÚC: Friendly
HƯỚNG DẪN AI: Ask a question for engagement
------------------------------------------"""
                        )
                        
                        (
                            sc_ns,
                            sc_gs,
                            sc_dn,
                            sc_sp,
                            sc_du,
                            sc_pp,
                            sc_po,
                        ) = _gen_settings()
                        sc_btn = gr.Button("Generate Script / Sinh giọng theo kịch bản", variant="primary")
                    with gr.Column(scale=1):
                        sc_parsed_markdown = gr.Markdown(label="Parsed Script Summary / Tóm tắt kịch bản đã phân tích")
                        
                        sc_audio1 = gr.Audio(label="Segment 1 Output / Kết quả phân đoạn 1", type="numpy")
                        sc_audio2 = gr.Audio(label="Segment 2 Output / Kết quả phân đoạn 2", type="numpy")
                        sc_audio3 = gr.Audio(label="Segment 3 Output / Kết quả phân đoạn 3", type="numpy")
                        sc_audio4 = gr.Audio(label="Segment 4 Output / Kết quả phân đoạn 4", type="numpy")
                        sc_audio5 = gr.Audio(label="Segment 5 Output / Kết quả phân đoạn 5", type="numpy")
                        
                        sc_zip = gr.File(label="Download All WAVs (ZIP) / Tải xuống tất cả các tệp (ZIP)")
                        sc_status = gr.Textbox(label="Status / Trạng thái", lines=5)

                def _script_clone_fn(
                    lang, 
                    ref_audio,
                    ref_text,
                    script_text,
                    ns, gs, dn, sp, du, pp, po
                ):
                    results = [None] * 5
                    statuses = []
                    
                    if not script_text or not script_text.strip():
                        return None, None, None, None, None, None, "", "Error: Script is empty."
                    
                    try:
                        segments = parse_script(script_text)
                    except Exception as e:
                        return None, None, None, None, None, None, "", f"Error parsing script: {e}"
                        
                    if not segments:
                        return None, None, None, None, None, None, "", "Error: No valid segments found in script. Please check the format."
                    
                    prompt = None
                    if ref_audio:
                        try:
                            # Pre-generate clone prompt
                            prompt = model.create_voice_clone_prompt(
                                ref_audio=ref_audio,
                                ref_text=ref_text or None,
                            )
                        except Exception as e:
                            return None, None, None, None, None, None, "", f"Error encoding reference audio: {e}"
                    
                    mode = "clone" if (ref_audio and str(ref_audio).strip()) else "design"
                    
                    temp_dir = tempfile.mkdtemp()
                    wav_paths = []
                    
                    parsed_summary = "### Parsed Segments:\n"
                    for idx, seg in enumerate(segments):
                        inst_display = f"*{seg['raw_instruct']}*" if seg['raw_instruct'] else ""
                        mapped_display = f" [AI tag: `{seg['valid_instruct']}`]" if seg['valid_instruct'] else ""
                        parsed_summary += f"- **Segment {seg['id']}** ({seg['duration']}s): {inst_display}{mapped_display} - \"{seg['text'][:30]}...\"\n"
                    
                    for idx, seg in enumerate(segments[:5]):
                        seg_id = seg["id"]
                        text = seg["text"]
                        duration_val = seg["duration"]
                        instruct_val = seg["valid_instruct"] # Use valid instruct mapped for OmniVoice
                        
                        try:
                            res, stat = _gen(
                                text,
                                lang,
                                ref_audio if (ref_audio and str(ref_audio).strip()) else None,
                                instruct_val,
                                ns,
                                gs,
                                dn,
                                sp,
                                duration_val,
                                pp,
                                po,
                                mode,
                                ref_text or None,
                                prompt,
                            )
                            results[idx] = res
                            statuses.append(f"Segment {seg_id}: {stat}")
                            
                            # Save to temp wave file for ZIP
                            if res and res[1] is not None:
                                import soundfile as sf
                                wav_path = os.path.join(temp_dir, f"segment_{seg_id}.wav")
                                sf.write(wav_path, res[1], res[0])
                                wav_paths.append(wav_path)
                                
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            statuses.append(f"Segment {seg_id}: Error: {e}")
                    
                    zip_path = None
                    if wav_paths:
                        zip_path = os.path.join(temp_dir, "all_segments.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for wp in wav_paths:
                                zipf.write(wp, os.path.basename(wp))
                                
                    return results[0], results[1], results[2], results[3], results[4], zip_path, parsed_summary, "\n".join(statuses)

                sc_btn.click(
                    _script_clone_fn,
                    inputs=[
                        sc_lang,
                        sc_ref_audio,
                        sc_ref_text,
                        sc_script,
                        sc_ns,
                        sc_gs,
                        sc_dn,
                        sc_sp,
                        sc_du,
                        sc_pp,
                        sc_po,
                    ],
                    outputs=[sc_audio1, sc_audio2, sc_audio3, sc_audio4, sc_audio5, sc_zip, sc_parsed_markdown, sc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
