"""Self-contained case selector; only the selected episode has active players."""
import html
import json
from pathlib import Path
import zipfile
from robouse.data.datasets import read_jsonl


def build(root):
    root = Path(root).resolve()
    episodes = read_jsonl(root / "episodes.jsonl")
    records = []
    files = set()
    for ep in episodes:
        videos = ep.get("videos", {})
        cameras = {"head", "left_wrist", "right_wrist"}
        if set(videos) - cameras:
            raise ValueError("Unknown camera name")
        for name in videos.values():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Missing original video")
            files.add(name)
        records.append(
            {
                "videos": videos,
                "missing_views": sorted(cameras - set(videos)),
                **{
                k: ep.get(k)
                for k in (
                    "episode_id",
                    "policy_version",
                    "score",
                    "termination_reason",
                    "timeline",
                )
                },
            }
        )
    payload = (
        json.dumps(records, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    # Only file-relative video paths and JSON embedded as inert text; no CDN.
    page = (
        """<meta charset="utf-8"><title>RoboUse results</title>
<style>body{font-family:system-ui;margin:24px}video{width:31%;vertical-align:top}pre{white-space:pre-wrap}select{max-width:100%}</style>
<h1>RoboUse results</h1><select id="cases"></select><p id="score"></p>
<video controls preload="none"></video><video controls preload="none"></video><video controls preload="none"></video>
<pre id="timeline"></pre><script id="data" type="application/json">"""
        + payload
        + """</script>
<script>
const cases=JSON.parse(document.getElementById('data').textContent), selector=document.getElementById('cases');
cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.episode_id+' | score='+c.score+' | '+c.termination_reason;selector.appendChild(o)});
function show(){const c=cases[Number(selector.value)];if(!c)return;
document.getElementById('score').textContent='Policy '+c.policy_version+' | score='+c.score+' | '+c.termination_reason+' | unavailable views: '+c.missing_views.join(', ');
document.querySelectorAll('video').forEach((v,i)=>{v.pause();v.removeAttribute('src');const name=['head','left_wrist','right_wrist'][i];v.title=name;if(c.videos[name])v.src=c.videos[name];v.load()});
document.getElementById('timeline').textContent=JSON.stringify(c.timeline,null,2)}selector.onchange=show;show();
</script>"""
    )
    (root / "index.html").write_text(page)
    with zipfile.ZipFile(root / "offline.zip", "w", zipfile.ZIP_STORED) as archive:
        metadata = [name for name in ("metrics.json", "run_manifest.json", "episode_metrics.json") if (root / name).is_file()]
        for name in ["index.html", "episodes.jsonl", *metadata, *sorted(files)]:
            archive.write(root / name, name)
