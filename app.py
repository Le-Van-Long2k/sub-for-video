
import asyncio
import os
import re
import gc
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

# =========================================================
# CONFIGURATION
# =========================================================

MODELS = {
    "Whisper Large-v3": "whisper:large-v3",
    "Whisper Large-v3 Turbo": "whisper:large-v3-turbo",
    "Whisper Medium": "whisper:medium",
    "Qwen3-ASR 1.7B": "qwen:1.7B",
    "Qwen3-ASR 0.6B": "qwen:0.6B",
    "Fun-ASR-Nano-2512": "funasr:nano",
    "SenseVoiceSmall": "funasr:sensevoice",
}

LANGUAGES = {
    "Tự động": None,
    "Tiếng Trung": "Chinese",
    "Tiếng Anh": "English",
    "Tiếng Nhật": "Japanese",
    "Tiếng Hàn": "Korean",
    "Tiếng Việt": "Vietnamese",
    "Tiếng Đức": "German",
}

ALIGN_LANGUAGES = {
    "Chinese", "English", "Cantonese", "French",
    "German", "Italian", "Japanese", "Korean",
    "Portuguese", "Russian", "Spanish"
}

qwen_cache = {}
whisper_cache = {}
funasr_cache = {}


# =========================================================
# UTILITIES
# =========================================================

def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
    except Exception:
        raise gr.Error(
            "Không tìm thấy FFmpeg. Hãy cài FFmpeg và thêm vào PATH."
        )


def extract_audio(video_path):
    temp_dir = tempfile.mkdtemp(prefix="video_srt_")
    audio_path = os.path.join(temp_dir, "audio.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])

    return audio_path, temp_dir


def format_time(seconds):
    ms = max(0, round(float(seconds) * 1000))

    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000

    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{seconds:02},"
        f"{milliseconds:03}"
    )


def create_srt(segments):
    output = []

    for segment in segments:
        text = str(segment["text"]).strip()

        if not text:
            continue

        start = float(segment["start"])
        end = max(start + 0.08, float(segment["end"]))

        index = len(output) + 1

        output.append(
            f"{index}\n"
            f"{format_time(start)} --> {format_time(end)}\n"
            f"{text}"
        )

    return "\n\n".join(output) + "\n"


def split_text(text, max_chars=42):
    text = re.sub(r"\s+", " ", str(text)).strip()

    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # First split at sentence punctuation. This works well for Chinese,
    # Japanese and other languages that normally do not use spaces.
    sentences = re.split(r"(?<=[.!?。！？；;])\s*", text)
    output = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            output.append(current.strip())
        current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            flush()
            # Keep word boundaries for space-separated languages;
            # otherwise split long CJK text by characters.
            units = sentence.split() if " " in sentence else list(sentence)
            for unit in units:
                separator = " " if " " in sentence else ""
                candidate = current + (separator if current else "") + unit
                if current and len(candidate) > max_chars:
                    flush()
                current = current + (separator if current else "") + unit
            flush()
        else:
            candidate = (current + " " + sentence).strip()
            if current and len(candidate) > max_chars:
                flush()
            current = (current + " " + sentence).strip()

    flush()
    return output


async def translate_segments_to_vietnamese(segments, progress):
    from googletrans import Translator

    max_batch_characters = 4000
    max_batch_segments = 50
    batches = []
    current_batch = []
    current_characters = 0

    for segment in segments:
        text = segment["text"]

        if current_batch and (
            current_characters + len(text) > max_batch_characters
            or len(current_batch) >= max_batch_segments
        ):
            batches.append(current_batch)
            current_batch = []
            current_characters = 0

        current_batch.append(segment)
        current_characters += len(text)

    if current_batch:
        batches.append(current_batch)

    translated_segments = []

    async with Translator() as translator:
        for batch_index, batch in enumerate(batches, start=1):
            translations = await translator.translate(
                [segment["text"] for segment in batch],
                dest="vi"
            )

            if len(translations) != len(batch):
                raise RuntimeError(
                    "Số câu dịch không khớp với số câu phụ đề."
                )

            for segment, translation in zip(batch, translations):
                translated_text = (translation.text or "").strip()

                if not translated_text:
                    raise RuntimeError(
                        "Google Translate trả về câu dịch rỗng."
                    )

                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text
                })

            progress(
                0.8 + 0.15 * batch_index / len(batches),
                desc=(
                    "Đang dịch phụ đề sang tiếng Việt "
                    f"({batch_index}/{len(batches)})..."
                )
            )

    return translated_segments


def create_video_playback_copy(video_path):
    descriptor, playback_path = tempfile.mkstemp(
        prefix="video_playback_",
        suffix=Path(video_path).suffix,
        dir=str(Path(video_path).parent)
    )
    os.close(descriptor)
    shutil.copyfile(video_path, playback_path)
    return playback_path


def group_aligned_words(items, max_chars=42, max_seconds=5.5):
    segments = []

    current_text = ""
    current_start = None
    current_end = None

    def flush():
        nonlocal current_text, current_start, current_end

        if current_text.strip() and current_start is not None:
            segments.append({
                "start": current_start,
                "end": max(current_start + 0.08, current_end),
                "text": current_text.strip()
            })

        current_text = ""
        current_start = None
        current_end = None

    for item in items:
        token = str(getattr(item, "text", "") or "").strip()

        if not token:
            continue

        start = float(getattr(item, "start_time", 0))
        end = float(getattr(item, "end_time", start + 0.1))

        needs_space = (
            current_text
            and token[0].isascii()
            and token[0].isalnum()
            and current_text[-1].isascii()
            and current_text[-1].isalnum()
        )

        addition = (" " if needs_space else "") + token

        should_split = (
            current_text
            and (
                len(current_text) + len(addition) > max_chars
                or end - current_start > max_seconds
                or re.search(r"[.!?。！？]$", current_text)
            )
        )

        if should_split:
            flush()

        if current_start is None:
            current_start = start

        current_text += addition
        current_end = end

    flush()

    return segments


# =========================================================
# QWEN MODEL
# =========================================================

def load_qwen(model_size, use_aligner, progress):

    import torch
    from qwen_asr import Qwen3ASRModel

    key = (model_size, use_aligner)

    if key in qwen_cache:
        return qwen_cache[key]

    model_id = (
        "Qwen/Qwen3-ASR-1.7B"
        if model_size == "1.7B"
        else "Qwen/Qwen3-ASR-0.6B"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    progress(0.2, desc=f"Loading {model_id}...")

    kwargs = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": 1,
        "max_new_tokens": 4096,
    }

    if use_aligner:
        kwargs["forced_aligner"] = (
            "Qwen/Qwen3-ForcedAligner-0.6B"
        )

        kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": device,
        }

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        **kwargs
    )

    qwen_cache[key] = model

    return model


# =========================================================
# FASTER WHISPER
# =========================================================

def load_whisper(model_size, progress):

    import ctranslate2
    from faster_whisper import WhisperModel

    try:
        cuda_device_count = ctranslate2.get_cuda_device_count()
    except RuntimeError as e:
        raise gr.Error(
            f"Faster-Whisper yêu cầu GPU CUDA, nhưng không thể khởi tạo CUDA: {e}"
        ) from e

    if cuda_device_count < 1:
        raise gr.Error(
            "Faster-Whisper bắt buộc dùng GPU CUDA, nhưng không phát hiện GPU CUDA."
        )

    if model_size in whisper_cache:
        return whisper_cache[model_size]

    progress(0.2, desc=f"Loading Whisper {model_size}...")

    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16",
        cpu_threads=max(2, (os.cpu_count() or 4) // 2)
    )

    whisper_cache[model_size] = model

    return model


# =========================================================
# FUNASR MODELS: FUN-ASR-NANO + SENSEVOICESMALL
# =========================================================

def load_funasr(model_kind, progress):
    """Load one FunASR model and cache it until the user unloads models."""
    import torch
    from funasr import AutoModel

    if model_kind in funasr_cache:
        return funasr_cache[model_kind]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = (
        "FunAudioLLM/Fun-ASR-Nano-2512"
        if model_kind == "nano"
        else "FunAudioLLM/SenseVoiceSmall"
    )

    progress(0.2, desc=f"Đang tải {model_id} trên {device}...")

    kwargs = {
        "model": model_id,
        "hub": "hf",
        "device": device,
    }
    if model_kind == "nano":
        # Fun-ASR-Nano uses the model's custom FunASR implementation.
        kwargs["trust_remote_code"] = True

    model = AutoModel(**kwargs)
    funasr_cache[model_kind] = model
    return model


def _funasr_result_text(result):
    """Normalize FunASR's slightly different result container shapes."""
    if isinstance(result, dict):
        return str(result.get("text", "") or "").strip()
    if isinstance(result, (list, tuple)):
        for item in result:
            text = _funasr_result_text(item)
            if text:
                return text
    return ""


def _audio_duration(audio_path):
    import wave
    with wave.open(audio_path, "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def _make_audio_chunks(audio_path, temp_dir, chunk_seconds=15):
    """Create short WAV chunks so text-only ASR models can yield usable SRT timing."""
    duration = _audio_duration(audio_path)
    chunks = []
    start = 0.0
    index = 0

    while start < duration:
        length = min(float(chunk_seconds), duration - start)
        chunk_path = os.path.join(temp_dir, f"asr_chunk_{index:05d}.wav")
        command = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", audio_path,
            "-t", f"{length:.3f}", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", chunk_path
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Không thể cắt audio chunk: {proc.stderr[-1000:]}")
        chunks.append((chunk_path, start, start + length))
        start += length
        index += 1

    return chunks


def transcribe_funasr(
    audio_path, model_kind, language, max_chars, max_seconds, temp_dir, progress
):
    """Transcribe FunASR in short chunks; SRT times are estimated per chunk.

    The official Fun-ASR-Nano model card currently lists timestamp output as
    TODO. SenseVoice's standard transcription result is also text-oriented,
    so this function does not pretend those timestamps are word-level timings.
    """
    model = load_funasr(model_kind, progress)

    if model_kind == "nano" and language not in (None, "Chinese", "English", "Japanese"):
        raise gr.Error(
            "Fun-ASR-Nano-2512 chỉ hỗ trợ tiếng Trung, tiếng Anh và tiếng Nhật. "
            "Hãy chọn một trong các ngôn ngữ này hoặc dùng Tự động."
        )

    fun_language = {
        "Chinese": "中文",
        "English": "英文",
        "Japanese": "日文",
    }.get(language)
    sense_language = {
        "Chinese": "zh", "English": "en", "Cantonese": "yue",
        "Japanese": "ja", "Korean": "ko",
    }.get(language, "auto")

    # Shorter chunks improve approximate subtitle placement. The user slider
    # controls subtitle length; chunks stay between 10 and 20 seconds.
    chunk_seconds = max(10, min(20, int(float(max_seconds) * 2)))
    chunks = _make_audio_chunks(audio_path, temp_dir, chunk_seconds)
    output = []

    for index, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
        progress(
            0.35 + 0.55 * (index / max(1, len(chunks))),
            desc=f"{('Fun-ASR-Nano' if model_kind == 'nano' else 'SenseVoice')} "
                 f"đang nhận dạng đoạn {index + 1}/{len(chunks)}..."
        )

        if model_kind == "nano":
            generate_kwargs = {
                "input": [chunk_path],
                "cache": {},
                "batch_size": 1,
                "itn": True,
            }
            if fun_language:
                generate_kwargs["language"] = fun_language
            result = model.generate(**generate_kwargs)
            raw_text = _funasr_result_text(result)
        else:
            result = model.generate(
                input=chunk_path,
                batch_size_s=30,
                language=sense_language,
                use_itn=True,
            )
            raw_text = _funasr_result_text(result)
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                raw_text = rich_transcription_postprocess(raw_text)
            except Exception:
                pass

        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        if not raw_text:
            continue

        pieces = split_text(raw_text, int(max_chars))
        if not pieces:
            continue

        # Text-only model: spread each chunk's duration over its text pieces.
        # These are approximate chunk-level timestamps, not forced alignment.
        weights = [max(1, len(piece)) for piece in pieces]
        total_weight = sum(weights)
        cursor = chunk_start
        chunk_duration = chunk_end - chunk_start

        for piece, weight in zip(pieces, weights):
            piece_duration = chunk_duration * weight / total_weight
            output.append({
                "start": cursor,
                "end": min(chunk_end, cursor + piece_duration),
                "text": piece,
            })
            cursor += piece_duration

    detected_language = language or "Tự động nhận diện"
    return output, detected_language


# =========================================================
# TRANSCRIPTION
# =========================================================

def language_code(language):

    codes = {
        "Chinese": "zh",
        "English": "en",
        "Japanese": "ja",
        "Korean": "ko",
        "Vietnamese": "vi",
        "Cantonese": "yue",
        "Thai": "th",
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Indonesian": "id",
    }

    return codes.get(language)


def transcribe_qwen(
    audio_path,
    model_size,
    language,
    use_aligner,
    max_chars,
    progress
):

    supported = language in ALIGN_LANGUAGES

    can_align = use_aligner and supported

    model = load_qwen(
        model_size,
        can_align,
        progress
    )

    progress(0.5, desc="Qwen3-ASR đang nhận dạng...")

    results = model.transcribe(
        audio=audio_path,
        language=language,
        return_time_stamps=can_align
    )

    if not results:
        raise RuntimeError("Qwen không trả về kết quả.")

    result = results[0]

    detected_language = result.language
    transcript = result.text.strip()

    if not transcript:
        raise RuntimeError("Không nhận dạng được lời thoại.")

    segments = []

    timestamps = result.time_stamps

    if timestamps is not None:
        items = getattr(timestamps, "items", None)

        if items:
            segments = group_aligned_words(
                items,
                max_chars=max_chars
            )

    return segments, detected_language, transcript


def transcribe_whisper(
    audio_path,
    model_size,
    language,
    max_chars,
    progress
):

    model = load_whisper(model_size, progress)

    progress(0.5, desc="Whisper đang nhận dạng...")

    segments_iter, info = model.transcribe(
        audio_path,
        language=language_code(language),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True
    )

    output = []

    for segment in segments_iter:

        text = (segment.text or "").strip()

        if not text:
            continue

        pieces = split_text(text, max_chars)

        duration = segment.end - segment.start

        weights = [max(1, len(p)) for p in pieces]
        total = sum(weights)

        cursor = segment.start

        for piece, weight in zip(pieces, weights):

            piece_duration = duration * weight / total

            output.append({
                "start": cursor,
                "end": min(segment.end, cursor + piece_duration),
                "text": piece
            })

            cursor += piece_duration

    return output, info.language


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_srt(
    video,
    model_choice,
    language_label,
    max_chars,
    max_seconds,
    use_aligner,
    progress=gr.Progress()
):

    if not video:
        raise gr.Error("Vui lòng tải video lên.")

    video_path = video if isinstance(video, str) else video.name

    if not os.path.isfile(video_path):
        raise gr.Error("Không tìm thấy video.")

    audio_path = None
    temp_dir = None

    try:

        check_ffmpeg()

        progress(0.05, desc="Đang tách âm thanh...")

        audio_path, temp_dir = extract_audio(video_path)

        language = LANGUAGES[language_label]

        model_type = MODELS[model_choice]

        if model_type.startswith("funasr:"):

            model_kind = model_type.split(":", 1)[1]
            segments, detected_language = transcribe_funasr(
                audio_path,
                model_kind,
                language,
                int(max_chars),
                float(max_seconds),
                temp_dir,
                progress
            )

            transcript = " ".join(seg["text"] for seg in segments)

        elif model_type.startswith("qwen:"):

            model_size = model_type.split(":")[1]

            segments, detected_language, transcript = transcribe_qwen(
                audio_path,
                model_size,
                language,
                use_aligner,
                int(max_chars),
                progress
            )

            # No forced alignment: do not fabricate accurate timestamps.
            if not segments:
                raise gr.Error(
                    "Qwen đã nhận dạng được văn bản nhưng chưa có "
                    "timestamp chính xác. Hãy bật ForcedAligner "
                    "với ngôn ngữ được hỗ trợ, hoặc chọn Faster-Whisper."
                )

        else:

            whisper_size = model_type.split(":")[1]

            segments, detected_language = transcribe_whisper(
                audio_path,
                whisper_size,
                language,
                int(max_chars),
                progress
            )

        if not segments:
            raise gr.Error("Không tạo được phụ đề.")

        # Limit excessively long subtitle durations by splitting text
        # while preserving the source segment's overall time range.
        final_segments = []

        for seg in segments:
            pieces = split_text(seg["text"], int(max_chars))
            duration = seg["end"] - seg["start"]
            weights = [max(1, len(p)) for p in pieces]
            total = sum(weights)
            cursor = seg["start"]

            for piece, weight in zip(pieces, weights):
                part_duration = duration * weight / total
                final_segments.append({
                    "start": cursor,
                    "end": min(seg["end"], cursor + part_duration),
                    "text": piece
                })
                cursor += part_duration

        srt_content = create_srt(final_segments)

        srt_path = str(
            Path(video_path).with_suffix(".srt")
        )

        translated_srt_path = str(
            Path(video_path).with_name(
                f"{Path(video_path).stem}.vi.srt"
            )
        )

        progress(0.8, desc="Đang dịch phụ đề sang tiếng Việt...")

        translated_segments = asyncio.run(
            translate_segments_to_vietnamese(final_segments, progress)
        )
        translated_srt_content = create_srt(translated_segments)

        Path(srt_path).write_text(
            srt_content,
            encoding="utf-8-sig"
        )

        Path(translated_srt_path).write_text(
            translated_srt_content,
            encoding="utf-8-sig"
        )

        progress(0.96, desc="Đang chuẩn bị phát video có phụ đề...")
        video_playback = gr.update(
            value=create_video_playback_copy(video_path),
            subtitles=[
                {
                    "text": segment["text"],
                    "timestamp": [segment["start"], segment["end"]]
                }
                for segment in translated_segments
            ]
        )

        status = (
            "HOÀN TẤT!\n\n"
            f"Model: {model_choice}\n"
            f"Ngôn ngữ: {detected_language}\n"
            f"Số đoạn: {len(final_segments)}\n"
            f"SRT gốc: {srt_path}\n"
            f"SRT tiếng Việt: {translated_srt_path}\n"
            "Video gốc sẽ phát trong ứng dụng kèm phụ đề tiếng Việt."
        )

        progress(1.0, desc="Hoàn thành!")

        return (
            srt_path,
            srt_content,
            translated_srt_path,
            video_playback,
            status
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(
            f"{type(e).__name__}: {e}"
        )

    finally:

        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# =========================================================
# MEMORY MANAGEMENT
# =========================================================

def unload_models():

    qwen_cache.clear()
    whisper_cache.clear()
    funasr_cache.clear()

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass

    return "Đã giải phóng cache model."


# =========================================================
# GRADIO UI
# =========================================================

CSS = """
.gradio-container {
    max-width: 1600px !important;
}
"""

with gr.Blocks(
    title="Video to SRT"
) as demo:

    gr.Markdown(
        """
        # 🎬 VIDEO TO SRT
        ### Local AI Speech Recognition

        Tạo phụ đề tự động từ video bằng nhiều mô hình AI.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            video_input = gr.File(
                label="Video đầu vào",
                file_types=["video"],
                type="filepath"
            )

            model_input = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Whisper Large-v3",
                label="Chọn mô hình ASR"
            )

            language_input = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Tự động",
                label="Ngôn ngữ"
            )

            with gr.Accordion("Cài đặt nâng cao", open=True):

                max_chars = gr.Slider(
                    minimum=20,
                    maximum=80,
                    value=42,
                    step=1,
                    label="Số ký tự tối đa mỗi dòng"
                )

                max_seconds = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=5.5,
                    step=0.5,
                    label="Thời lượng tối đa mỗi đoạn (giây)"
                )

                use_aligner = gr.Checkbox(
                    value=True,
                    label="Qwen: sử dụng ForcedAligner để căn timestamp"
                )

            run_button = gr.Button(
                "🚀 TẠO PHỤ ĐỀ",
                variant="primary"
            )

            unload_button = gr.Button(
                "🧹 Giải phóng RAM / VRAM"
            )

            status_output = gr.Textbox(
                label="Trạng thái",
                lines=6
            )

        with gr.Column(scale=1):
            file_output = gr.File(
                label="Tải phụ đề SRT"
            )

            translated_file_output = gr.File(
                label="Tải phụ đề tiếng Việt (.vi.srt)"
            )

            srt_output = gr.Textbox(
                label="Nội dung SRT",
                lines=8,
                max_lines=15,
                interactive=False
            )

        with gr.Column(scale=1):

            translated_video_output = gr.Video(
                label="Phát lại video gốc có phụ đề tiếng Việt",
                interactive=False
            )

    run_button.click(
        fn=generate_srt,
        inputs=[
            video_input,
            model_input,
            language_input,
            max_chars,
            max_seconds,
            use_aligner
        ],
        outputs=[
            file_output,
            srt_output,
            translated_file_output,
            translated_video_output,
            status_output
        ]
    )

    unload_button.click(
        fn=unload_models,
        outputs=status_output
    )

    gr.Markdown(
        """
        ---
        Bản dịch tiếng Việt dùng Google Translate, cần kết nối Internet.
        Video phát lại dùng video gốc và phụ đề phủ theo thời gian; không xuất MP4 mới.

        **Thông tin:**
        - Model được tải tự động từ Hugging Face khi sử dụng lần đầu.
        - Faster-Whisper cung cấp timestamp theo đoạn.
        - Qwen3-ASR có thể dùng ForcedAligner để căn timestamp chi tiết.
        - Fun-ASR-Nano và SenseVoice không trả timestamp chi tiết trong pipeline này;
          app chia audio thành đoạn ngắn và ước lượng timestamp cho SRT.
        - Fun-ASR-Nano hỗ trợ tiếng Trung, tiếng Anh và tiếng Nhật.
        """
    )


if __name__ == "__main__":

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS
    )
