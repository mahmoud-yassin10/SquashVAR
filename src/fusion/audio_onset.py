import argparse
import numpy as np
import librosa
import soundfile as sf  # ensures libsndfile is present for various formats

def detect_onsets(
    wav_path: str,
    sr_target: int = 22050,
    backtrack: bool = True,
    hop_length: int = 512,
    pre_max: int = 20,
    post_max: int = 20,
    pre_avg: int = 100,
    post_avg: int = 100,
    delta: float = 0.2,
):
    """
    Return onset times (seconds) from an audio file.
    Defaults are conservative for hard, percussive impacts (e.g., squash front-wall hits).
    """
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    on_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
        backtrack=backtrack,
        hop_length=hop_length,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
    )
    on_times = librosa.frames_to_time(on_frames, sr=sr, hop_length=hop_length)
    return np.asarray(on_times, dtype=float)

def shifted_onsets(onsets_sec: np.ndarray, offset_ms: float = 0.0):
    """Apply a constant AV sync offset (ms) to onset times."""
    if onsets_sec is None or len(onsets_sec) == 0:
        return np.asarray([], dtype=float)
    return onsets_sec + (offset_ms / 1000.0)

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="path to audio wav")
    ap.add_argument("--offset_ms", type=float, default=0.0, help="shift onsets by constant ms")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--delta", type=float, default=0.2)
    a = ap.parse_args()
    on = detect_onsets(a.wav, sr_target=a.sr, delta=a.delta)
    on = shifted_onsets(on, a.offset_ms)
    print("onsets_sec:", np.round(on, 4).tolist())

if __name__ == "__main__":
    _cli()
