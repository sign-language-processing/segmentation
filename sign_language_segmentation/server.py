import gzip
import json
import os
import traceback
from datetime import datetime, UTC
from pathlib import Path

from flask import Flask, request, abort, make_response, jsonify
from pose_format import Pose

from sign_language_segmentation.bin import segment_pose

app = Flask(__name__)

CACHE_TTL = 86400  # 1 day in seconds


def resolve_path(uri: str):
    # Map gs:// URIs to the gcsfuse mount point, or return as-is
    return uri.replace("gs://", "/mnt/")


def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def after_request(response):
    return add_cors(response)


@app.errorhandler(Exception)
def handle_exception(e):
    print("Exception", e)
    traceback.print_exc()

    code = e.code if hasattr(e, "code") else 500
    message = str(e)
    print("HTTP exception", code, message)

    return make_response(jsonify(message=message, code=code), code)


def load_pose(uri: str) -> Pose:
    pose_file_path = Path(resolve_path(uri))
    if not pose_file_path.exists():
        raise FileNotFoundError(f"File does not exist: {uri}")
    with pose_file_path.open("rb") as f:
        return Pose.read(f)


def tiers_to_seconds(tiers: dict, fps: float) -> dict:
    """Convert frame-index segment dicts to seconds."""
    return {
        tier: [{"start": round(seg["start"] / fps, 4), "end": round(seg["end"] / fps, 4)}
               for seg in segments]
        for tier, segments in tiers.items()
    }


def gzip_json(data: dict):
    body = json.dumps(data, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(body)
    response = make_response(compressed, 200)
    response.headers["Content-Type"] = "application/json"
    response.headers["Content-Encoding"] = "gzip"
    return response


@app.route('/health', methods=['GET'])
def health_check():
    body = {
        'status': 'healthy',
        'timestamp': datetime.now(tz=UTC).isoformat(),
        'service': 'segmentation',
    }
    return make_response(jsonify(body), 200)


@app.route("/segments", methods=['GET', 'OPTIONS'])
def get_segments():
    if request.method == 'OPTIONS':
        return make_response("", 204)

    pose_uri = request.args.get("pose")
    if not pose_uri:
        abort(make_response(jsonify(message="Missing `pose` query parameter"), 400))

    pose = load_pose(pose_uri)

    if len(pose.body.data) == 1:
        return gzip_json({"sign": [], "sentence": []})

    _eaf, tiers = segment_pose(pose)
    result = tiers_to_seconds(tiers, pose.body.fps)

    response = gzip_json({"sign": result["SIGN"], "sentence": result["SENTENCE"]})
    response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL}"
    return response


@app.route("/", methods=['POST'])
def pose_segmentation():
    body = request.get_json()
    for param in ['input', 'output']:
        if param not in body:
            abort(make_response(jsonify(message=f"Missing `{param}` body property"), 400))

    output_file_path = Path(resolve_path(body["output"]))
    if output_file_path.exists():
        return make_response(jsonify(message="Output file already exists", path=body["output"]), 208)

    pose = load_pose(body["input"])

    if len(pose.body.data) == 1:
        return make_response(jsonify(message="Pose has only one frame, no segmentation needed", path=body["output"]), 200)

    eaf, _tiers = segment_pose(pose)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving .eaf to disk ...")
    eaf.to_file(output_file_path)

    return make_response(jsonify(
        message="Pose segmentation completed successfully",
        path=body["output"],
    ), 200)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
