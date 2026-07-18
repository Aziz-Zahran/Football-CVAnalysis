# Football-CVAnalysis

Computer vision system for analyzing football match footage. Detects and tracks players, referees, and the ball across video frames using a fine-tuned YOLO model.

**Status: work in progress.** The detection and tracking pipeline runs, but annotation output and the actual analysis layer are not built yet.

## What it does right now

- Reads match video frame by frame
- Runs detection with a YOLO model trained on the Roboflow football-players-detection dataset
- Maps goalkeeper detections to the player class so they get tracked together
- Assigns persistent track IDs with ByteTrack via `supervision`
- Caches tracking results to a pickle stub so you don't re-run inference every time
- Writes the frames back out to video

## Layout

```
main.py              entry point
trackers/tracker.py  detection + ByteTrack wrapper
utils/video_utils.py video read/write
training/            training notebook and dataset
models/              trained weights
```

## Running it

```bash
uv sync
python main.py
```

You need `models/best.pt` and a clip in `input_videos/`. Weights, videos, datasets, and training runs are all gitignored, so grab them separately. The dataset comes from Roboflow and the notebook in `training/` covers the fine-tuning.

## Known issues

- Track IDs are unstable. A 750 frame clip with 22 players on screen produces IDs up to 77, so players are getting reassigned on occlusion
- The ball is only detected in about half the frames (366 of 750), so possession logic will need interpolation to fill the gaps

## Next up

- Draw bounding boxes and track IDs onto output frames
- Team assignment by jersey color
- Ball possession per player
- Camera movement compensation and perspective transform for real world coordinates
- Speed and distance covered metrics
