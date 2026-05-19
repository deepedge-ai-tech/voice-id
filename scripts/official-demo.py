# Copyright (c) Microsoft
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
import wespeaker

STYLE = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" integrity="sha256-YvdLHPgkqJ8DVUxjjnGVlMMJtNimJ6dYkowFFvp4kKs=" crossorigin="anonymous">
"""
OUTPUT_OK = (STYLE + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">The speakers are</h1></div>
        <div class="row"><h1 class="display-1 text-success" style="text-align: center">{:.1f}%</h1></div>
        <div class="row"><h1 style="text-align: center">similar</h1></div>
        <div class="row"><h1 class="text-success" style="text-align: center">Welcome, human!</h1></div>
        <div class="row"><small style="text-align: center">(You must get at least 70% to be considered the same person)</small></div>
    </div>
""")
OUTPUT_FAIL = (STYLE + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">The speakers are</h1></div>
        <div class="row"><h1 class="display-1 text-danger" style="text-align: center">{:.1f}%</h1></div>
        <div class="row"><h1 style="text-align: center">similar</h1></div>
        <div class="row"><h1 class="text-danger" style="text-align: center">Warning! stranger!</h1></div>
        <div class="row"><small style="text-align: center">(You must get at least 70% to be considered the same person)</small></div>
    </div>
""")

OUTPUT_ERROR = (STYLE + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">Input Error</h1></div>
        <div class="row"><h1 class="text-danger" style="text-align: center">{}!</h1></div>
    </div>
""")

cn_model = wespeaker.load_model("chinese")
en_model = wespeaker.load_model("english")

# Asset directory root (relative to Voice-ID project root)
ASSET = "asset"


def speaker_verification(audio_path1, audio_path2, lang="CN"):
    if audio_path1 is None or audio_path2 is None:
        return OUTPUT_ERROR.format("Please enter two audios")
    model = {"EN": en_model, "CN": cn_model}.get(lang)
    if model is None:
        return OUTPUT_ERROR.format("Please select a language")
    cos_score = model.compute_similarity(audio_path1, audio_path2)
    template = OUTPUT_OK if cos_score >= 0.70 else OUTPUT_FAIL
    return template.format(cos_score * 100)


# input
inputs = [
    gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speaker #1"),
    gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speaker #2"),
    gr.Radio(["EN", "CN"], label="Language", value="CN"),
]

output = gr.HTML()

# description
description = ("<p>WeSpeaker Demo ! Try it with your own voice ! Note: We recommend that the audio length be greater than 5s !</p>")

article = (
    "<p style='text-align: center'>"
    "<a href='https://github.com/wenet-e2e/wespeaker' target='_blank'>Github: Learn more about WeSpeaker</a>"
    "</p>")

A = ASSET
examples = [
    # ── 同人验证：测试音频 vs 注册音频（期望 > 70%） ──
    [f"{A}/john/test_segments/test_segment_000_4s.wav",   f"{A}/john/registration_segments/segment_000.wav",   "CN"],
    [f"{A}/john_d_usb/test_segments/test_segment_000_4s.wav", f"{A}/john_d_usb/registration_segments/segment_000.wav", "CN"],
    [f"{A}/john_metting_room/test_segments/test_segment_000_4s.wav", f"{A}/john_metting_room/registration_segments/segment_000.wav", "CN"],
    [f"{A}/frank/test_segments/test_segment_000_4s.wav",  f"{A}/frank/registration_segments/segment_01.wav",  "CN"],
    [f"{A}/michael/测试.wav",                              f"{A}/michael/registration_segments/segment_000.wav", "CN"],
    [f"{A}/qingqing/test_segments/test_segment_000_4s.wav", f"{A}/qingqing/registration_segments/segment_000.wav", "CN"],
    [f"{A}/xixi/test_segments/test_segment_000_4s.wav",    f"{A}/xixi/registration_segments/segment_000.wav",   "CN"],
    [f"{A}/zhong/test_segments/test_segment_000_4s.wav",   f"{A}/zhong/registration_segments/segment_000.wav",  "CN"],
    [f"{A}/zhong_old/测试.wav",                            f"{A}/zhong_old/registration_segments/segment_000.wav", "CN"],
    # ── 异人验证：不同人比对（期望 < 70%） ──
    [f"{A}/john/test_segments/test_segment_000_4s.wav",   f"{A}/frank/test_segments/test_segment_000_4s.wav",  "CN"],
    [f"{A}/john/test_segments/test_segment_000_4s.wav",   f"{A}/michael/测试.wav",                              "CN"],
    [f"{A}/qingqing/test_segments/test_segment_000_4s.wav", f"{A}/zhong/test_segments/test_segment_000_4s.wav", "CN"],
    [f"{A}/xixi/test_segments/test_segment_000_4s.wav",    f"{A}/frank/test_segments/test_segment_000_4s.wav",  "CN"],
    [f"{A}/michael/测试.wav",                              f"{A}/xixi/测试.wav",                                "CN"],
    # ── 混合音频测试 ──
    [f"{A}/john/john_mixed_qingqing_20pct.wav",            f"{A}/john/registration_segments/segment_000.wav",   "CN"],
    [f"{A}/frank/frank_mixed_zhong_20pct.wav",             f"{A}/frank/registration_segments/segment_01.wav",  "CN"],
]

interface = gr.Interface(
    fn=speaker_verification,
    inputs=inputs,
    outputs=output,
    title="Speaker Verification in WeSpeaker : 基于 WeSpeaker 的说话人确认",
    description=description,
    article=article,
    examples=examples,
)

interface.launch()