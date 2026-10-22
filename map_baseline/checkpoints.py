"""Save diagnostic snapshots without evaluating or consuming training RNG."""
from pathlib import Path


class SnapshotSchedule:
    def __init__(self, out, every, episodes):
        if every < 0 or episodes < 1:
            raise ValueError('Snapshot interval must be nonnegative and episode budget positive')
        self.directory = Path(out) / 'snapshots'
        self.points = sorted({0, episodes, *range(every, episodes, every)}) if every else []
        self.next_index = 0
        if self.points:
            self.directory.mkdir(parents=True, exist_ok=True)

    def capture(self, completed, writer):
        # Vector environments finish their current step before a snapshot.
        # Checkpoint metadata retains the actual completed episode count.
        while self.next_index < len(self.points) and self.points[self.next_index] <= completed:
            point = self.points[self.next_index]
            destination = self.directory / f'episodes_{point:06d}.pt'
            temporary = destination.with_suffix('.tmp')
            writer(temporary)
            temporary.replace(destination)
            self.next_index += 1
