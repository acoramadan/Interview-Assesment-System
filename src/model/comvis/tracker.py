from .utils import iou, now_ts
from .track import Track
from .config import IOU_THRESH, MAX_MISSED_SEC

class Tracker:
    def __init__(self): self.tracks = []

    def update(self, bboxes):
        assigned = set()

        for t in self.tracks:
            best, idx = 0.0, -1

            for j, b in enumerate(bboxes):

                if j in assigned: continue
                i = iou(t.bbox, b)

                if i > best: best, idx = i, j

            if best >= IOU_THRESH and idx >= 0:
                t.update_bbox(bboxes[idx]); assigned.add(idx)

        for j, b in enumerate(bboxes):
            if j not in assigned: self.tracks.append(Track(b))

        tnow = now_ts()
        self.tracks = [t for t in self.tracks if (tnow - t.ts_last) <= MAX_MISSED_SEC]
        return self.tracks