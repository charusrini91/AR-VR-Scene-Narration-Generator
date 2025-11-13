"""
AR-VR Scene Narration Generator
Single-file Python project (Streamlit UI + offline/local or API-backed image & TTS options)

Requirements (pip):
  pip install streamlit moviepy pillow pydub srt gTTS diffusers transformers torch accelerate safetensors

Notes:
- Image generation: by default uses Hugging Face diffusers (Stable Diffusion). If you don't have GPU, set use_api=True and provide an API-based function (example placeholder included).
- TTS: defaults to gTTS (requires internet). Optionally use pyttsx3 for offline TTS (commented). ElevenLabs support is sketched (requires API key and requests).
- This script produces: images, narration audio (.mp3), subtitles (.srt), preview video (.mp4), and a ZIP package ready for Unity import.
- Replace placeholders (API keys, model IDs) as needed.

Run (development):
  streamlit run AR-VR_Scene_Narration_Generator.py

"""

import os
import io
import sys
import json
import zipfile
import tempfile
from datetime import timedelta
from typing import List, Dict

# UI
import streamlit as st
from PIL import Image

# Optional image generation libs
try:
    import torch
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False

# TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# moviepy for preview
try:
    from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

# pydub for audio operations (optional)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# ---------- Utility functions ----------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------- Prompt expansion (simple) ----------
def expand_scene_prompt(prompt: str) -> (str, List[Dict]):
    """Expand a short prompt into narration text and a shot list.
    For production, replace with an LLM call to OpenAI/HF for richer output.
    """
    narration = (
        f"This scene is about: {prompt}.\n"
        "Observe the environment closely: the colors, the lighting, and the interaction between subjects.\n"
        "We pause to notice details in three camera angles."
    )
    # Generate a few shot prompts
    shot_list = [
        {"name": "wide", "prompt": f"{prompt}, cinematic wide shot, ultra-detailed, 8k"},
        {"name": "left_close", "prompt": f"{prompt}, left-side close-up, focus on details, cinematic"},
        {"name": "right_close", "prompt": f"{prompt}, right-side close-up, shallow depth of field, cinematic"},
    ]
    return narration, shot_list


# ---------- Image generation (Diffusers local) ----------
class ImageGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = None, use_api: bool = False):
        self.use_api = use_api
        self.model_id = model_id
        # Fixed device detection logic: ensure 'torch' is available before calling torch.cuda.is_available()
        if device is None:
            self.device = "cuda" if ('torch' in globals() and hasattr(torch, 'cuda') and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        self.pipe = None
        if not use_api and DIFFUSERS_AVAILABLE:
            try:
                # load pipeline (memory heavy)
                torch_dtype = getattr(torch, 'float16') if self.device == "cuda" and hasattr(torch, 'float16') else getattr(torch, 'float32')
                self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
                # Move to device if CUDA is available
                try:
                    self.pipe = self.pipe.to(self.device)
                except Exception:
                    # Some environments may not support .to with 'cpu' str; ignore
                    pass
            except Exception as e:
                st.warning(f"Could not load diffusers pipeline locally: {e}")
                self.pipe = None

    def generate(self, prompt: str, out_path: str, num_inference_steps: int = 20):
        """Generate a single image and save to out_path.
        If use_api=True, this should call a hosted API instead (placeholder below).
        """
        if self.use_api:
            # Placeholder for API-based image generation. The user should implement this.
            # Example: call Stability.ai or Replicate or OpenAI image API and save response bytes to out_path.
            raise NotImplementedError("API-based image generation not implemented. Set use_api=False or implement API call.")

        if self.pipe is None:
            raise RuntimeError("Local diffusers pipeline not initialized. Install diffusers + torch and have enough RAM/GPU.")

        image = self.pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        image.save(out_path)
        return out_path


# ---------- TTS generation ----------

def generate_tts_gtts(text: str, out_file: str, lang: str = "en") -> str:
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed. Install via `pip install gTTS`.")
    tts = gTTS(text, lang=lang)
    tts.save(out_file)
    return out_file


# Optional: pyttsx3 offline TTS (uncomment and install if desired)
# import pyttsx3
# def generate_tts_pyttsx3(text: str, out_file: str):
#     engine = pyttsx3.init()
#     engine.save_to_file(text, out_file)
#     engine.runAndWait()
#     return out_file

# Optional: ElevenLabs example (requires requests and API key)
# def generate_tts_elevenlabs(text: str, out_file: str, api_key: str, voice: str = 'alloy'):
#     # User must implement using ElevenLabs TTS endpoints
#     pass


# ---------- SRT creation ----------

def make_srt_from_text(text: str, out_srt: str, words_per_sec: float = 2.6):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    current_start = 0.0
    i = 1
    lines = []
    for s in sentences:
        words = s.split()
        duration = max(1.0, len(words) / words_per_sec)
        start = timedelta(seconds=current_start)
        end = timedelta(seconds=current_start + duration)
        # Format to HH:MM:SS,mmm
        def fmt(td):
            total_seconds = int(td.total_seconds())
            ms = int((td.total_seconds() - total_seconds) * 1000)
            hh = total_seconds // 3600
            mm = (total_seconds % 3600) // 60
            ss = total_seconds % 60
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
        block = f"{i}\n{fmt(start)} --> {fmt(end)}\n{s}.\n\n"
        lines.append(block)
        current_start += duration
        i += 1
    with open(out_srt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_srt


# ---------- Preview video assembly ----------

def make_preview_video(image_paths: List[str], audio_path: str, out_video: str, duration_per_image: float = 4.0):
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("moviepy not installed. Install via `pip install moviepy`.")
    clips = []
    for p in image_paths:
        clips.append(ImageClip(p).set_duration(duration_per_image))
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path)
    # Extend video if shorter than audio
    if audio.duration > video.duration:
        extra = audio.duration - video.duration
        last = ImageClip(image_paths[-1]).set_duration(extra)
        video = concatenate_videoclips([video, last], method="compose")
    video = video.set_audio(audio)
    # Write with basic logging disabled for cleaner output
    video.write_videofile(out_video, fps=24)
    return out_video


# ---------- Packaging for Unity ----------

def create_metadata_json(prompt: str, narration_file: str, shots: List[Dict], out_file: str):
    meta = {"prompt": prompt, "narration": os.path.basename(narration_file), "shots": shots}
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_file


def package_for_unity(out_zip: str, files: List[str], base_dir: str = None):
    with zipfile.ZipFile(out_zip, 'w') as z:
        for f in files:
            arcname = os.path.basename(f) if base_dir is None else os.path.relpath(f, base_dir)
            z.write(f, arcname=arcname)
    return out_zip


# ---------- Streamlit UI & orchestration ----------

def run_streamlit_app():
    st.set_page_config(page_title="AR-VR Scene Narration Generator", layout='wide')

    st.title("AR-VR Scene Narration Generator")
    with st.sidebar:
        st.markdown("## Settings")
        model_id = st.text_input("Diffusion Model ID (hf model path)", value="runwayml/stable-diffusion-v1-5")
        use_api = st.checkbox("Use API for image generation (not implemented)", value=False)
        tts_method = st.selectbox("TTS Method", options=("gTTS", "pyttsx3 (offline, if available)"))
        steps = st.slider("Diffusion steps", min_value=10, max_value=50, value=20)
        duration_per_image = st.slider("Preview seconds per image", min_value=2, max_value=8, value=4)

    prompt = st.text_area("Enter scene prompt", height=120)
    submit = st.button("Generate Scene Assets")

    if submit and prompt.strip():
        status = st.empty()
        status.info("Expanding prompt and preparing assets...")

        # Expand prompt
        narration_text, shot_list = expand_scene_prompt(prompt)

        # Create working directories
        out_root = os.path.join("generated_scenes", sanitize_filename(prompt)[:64])
        ensure_dir(out_root)
        images_dir = os.path.join(out_root, "images")
        ensure_dir(images_dir)
        audio_dir = os.path.join(out_root, "audio")
        ensure_dir(audio_dir)

        # Image generation
        status.info("Initializing image generator...")
        img_gen = ImageGenerator(model_id=model_id, use_api=use_api)

        image_paths = []
        status.info("Generating images (this may take time)...")
        for shot in shot_list:
            fname = f"{shot['name']}.png"
            out_path = os.path.join(images_dir, fname)
            try:
                img_gen.generate(shot['prompt'], out_path, num_inference_steps=steps)
                image_paths.append(out_path)
                st.image(out_path, caption=shot['name'])
            except Exception as e:
                st.error(f"Image generation failed for shot {shot['name']}: {e}")

        # TTS generation
        status.info("Generating narration (TTS)...")
        narration_file = os.path.join(audio_dir, "narration.mp3")
        try:
            if tts_method == "gTTS":
                generate_tts_gtts(narration_text, narration_file)
            else:
                # Fallback: attempt gTTS
                if GTTS_AVAILABLE:
                    generate_tts_gtts(narration_text, narration_file)
                else:
                    st.warning("pyttsx3 not implemented here; falling back to gTTS (if available).")
            st.audio(narration_file)
        except Exception as e:
            st.error(f"TTS generation failed: {e}")

        # SRT generation
        status.info("Creating subtitles (SRT)...")
        srt_file = os.path.join(out_root, "narration.srt")
        try:
            make_srt_from_text(narration_text, srt_file)
            st.markdown("Subtitles created.")
            with open(srt_file, 'r', encoding='utf-8') as f:
                st.text(f.read())
        except Exception as e:
            st.error(f"SRT creation failed: {e}")

        # Preview video
        status.info("Assembling preview video...")
        preview_video = os.path.join(out_root, "preview.mp4")
        try:
            if image_paths and os.path.exists(narration_file):
                make_preview_video(image_paths, narration_file, preview_video, duration_per_image=duration_per_image)
                st.video(preview_video)
            else:
                st.warning("Missing images or audio; preview video skipped.")
        except Exception as e:
            st.error(f"Preview video creation failed: {e}")

        # Metadata + package
        status.info("Packaging assets for Unity...")
        metadata = create_metadata_json(prompt, narration_file, shot_list, os.path.join(out_root, "metadata.json"))
        files_to_zip = image_paths + [narration_file, srt_file, metadata]
        zip_path = os.path.join(out_root, "unity_package.zip")
        try:
            package_for_unity(zip_path, files_to_zip, base_dir=out_root)
            with open(zip_path, 'rb') as f:
                st.download_button("Download Unity package (ZIP)", data=f, file_name=os.path.basename(zip_path))
            status.success("Generation complete. Package ready for Unity import.")
        except Exception as e:
            st.error(f"Packaging failed: {e}")


# ---------- Helpers ----------

def sanitize_filename(s: str) -> str:
    import re
    return re.sub(r'[^A-Za-z0-9 _\\-]', '_', s)


# ---------- Tests (lightweight) ----------

def run_basic_tests():
    """Run a few quick smoke tests that don't require heavy deps (diffusers, moviepy).
    This function is invoked only when running the script with `--test`.
    """
    print("Running basic tests...")
    # Test expand_scene_prompt
    prompt = "A futuristic classroom with robots"
    narration, shots = expand_scene_prompt(prompt)
    assert isinstance(narration, str) and len(narration) > 0, "Narration should be non-empty"
    assert isinstance(shots, list) and len(shots) >= 1, "Shot list should be non-empty"
    print("expand_scene_prompt OK")

    # Test sanitize_filename
    fname = sanitize_filename(prompt)
    assert all(c not in fname for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']), "sanitize_filename failed"
    print("sanitize_filename OK")

    # Test SRT generation (writes to temp file)
    tmpdir = tempfile.mkdtemp()
    srt = os.path.join(tmpdir, 'test.srt')
    make_srt_from_text(narration, srt)
    assert os.path.exists(srt), "SRT file not created"
    size = os.path.getsize(srt)
    assert size > 0, "SRT file is empty"
    print("make_srt_from_text OK")

    # Test metadata + packaging (with small dummy files)
    dummy_img = os.path.join(tmpdir, 'img.png')
    Image.new('RGB', (64,64), color=(73,109,137)).save(dummy_img)
    dummy_audio = os.path.join(tmpdir, 'audio.mp3')
    with open(dummy_audio, 'wb') as f:
        f.write(b'ID3')
    metadata = os.path.join(tmpdir, 'metadata.json')
    create_metadata_json(prompt, dummy_audio, shots, metadata)
    zipf = os.path.join(tmpdir, 'package.zip')
    package_for_unity(zipf, [dummy_img, dummy_audio, metadata], base_dir=tmpdir)
    assert os.path.exists(zipf), "ZIP package not created"
    print("Packaging OK")
    print("All basic tests passed.")


# ---------- Main ----------
if __name__ == "__main__":
    # If launched with `python script.py --test` run lightweight tests and exit
    if '--test' in sys.argv:
        run_basic_tests()
        sys.exit(0)

    # If launched normally, recommend using streamlit
    st_cli = os.environ.get('STREAMLIT_SERVER_RUN', None)
    if st_cli is None:
        # likely running as script; start streamlit via command line recommended
        print("This script is intended to be run with Streamlit: `streamlit run AR-VR_Scene_Narration_Generator.py`")
        print("Starting Streamlit UI... (if running inside Streamlit this message won't show)")
    run_streamlit_app()
