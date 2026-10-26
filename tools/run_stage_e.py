import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from types import SimpleNamespace
import av
from voxforge.audit.av_sync_qa import audit_av_sync
from voxforge.models import EpisodeScript, ChapterScript, DialogueTurn, ShowSpec
d = Path(sys.argv[1])
raw = json.load(open(d/"episode_script.json"))
def _mk(cls, dd): return cls(**{k: dd[k] for k in [f.name for f in dataclasses.fields(cls)] if k in dd})
chapters = []
for ch in raw["chapters"]:
    chd = _mk(ChapterScript, {**ch, "turns": []})
    chd.turns = [_mk(DialogueTurn, t) for t in ch["turns"]]
    chapters.append(chd)
script = _mk(EpisodeScript, {**raw, "chapters": chapters})
if isinstance(raw.get("spec"), dict): script.spec = _mk(ShowSpec, raw["spec"])
c = av.open(str(d/"documentary_film.mp4")); dur = c.duration/av.time_base; c.close()
audio_meta = SimpleNamespace(duration_sec=dur)
st = audit_av_sync(d/"documentary_film.mp4", d/"scene_plan.json", d/"captions.srt", script, audio_meta)
print("score:", st.final_score, "passed:", st.passed)
for c in st.checks: print("  ", c.check_id, c.measured_value, "/", c.target_spec)
p = json.load(open(d/"scene_plan.json"))
bad = [e for e in p["scenes"] if not str(e["validation_status"]).startswith(("validated","skipped"))]
print("mismatched:", len(bad))
print("subject mismatched:", p.get("stage_e", {}).get("subject_mismatches", 0))
for e in bad[:8]: print("  ", e["seg_index"], e["visual_family"], e["validation_status"], e["validation_note"][:90], "|", e["narration_clause"][:50])
