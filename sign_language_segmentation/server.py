import os
import traceback
from pathlib import Path

from flask import Flask, request, abort, make_response, jsonify
from pose_format import Pose

from sign_language_segmentation.bin import segment_pose

app = Flask(__name__)


def resolve_path(uri: str):
    # Map gs:// URIs to the gcsfuse mount point, or return as-is
    return uri.replace("gs://", "/mnt/")


@app.errorhandler(Exception)
def handle_exception(e):
    print("Exception", e)
    traceback.print_exc()

    code = e.code if hasattr(e, "code") else 500
    message = str(e)
    print("HTTP exception", code, message)

    return make_response(jsonify(message=message, code=code), code)


@app.route("/", methods=['POST'])
def pose_segmentation():
    # Get request parameters
    body = request.get_json()
    for param in ['input', 'output']:
        if param not in body:
            abort(make_response(jsonify(message=f"Missing `{param}` body property"), 400))

    # Check if output file already exists
    output_file_path = Path(resolve_path(body["output"]))
    if output_file_path.exists():
        return make_response(jsonify(message="Output file already exists", path=body["output"]), 208)

    # Check if input file exists at all
    pose_file_path = Path(resolve_path(body["input"]))
    if not pose_file_path.exists():
        raise Exception("File does not exist")

    with pose_file_path.open("rb") as f:
        pose = Pose.read(f)

    eaf, tiers = segment_pose(pose)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving .eaf to disk ...")
    eaf.to_file(output_file_path)

    return make_response(jsonify(
        message="Pose segmentation completed successfully",
        path=body["output"],
    ), 200)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
