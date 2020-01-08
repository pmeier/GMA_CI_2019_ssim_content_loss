import sys
import io
from contextlib import redirect_stdout, contextmanager
from collections import namedtuple, defaultdict
import re
import numpy as np
import pandas as pd

Event = namedtuple("event", ("step", "name", "value"))


class OperatorScoreParser:
    pattern = re.compile(
        "\s+Step\s+(?P<step>\d+)\s+\|(?P<operator>[\sa-zA-Z\-_()]+)(?P<score>[\d.+-eE]+)\s+"
    )

    def parse(self, line):
        match = self.pattern.match(line)
        if match is None:
            return None

        matches = [match.group(group) for group in ("step", "operator", "score")]
        step = int(matches[0])
        name = matches[1].strip()
        value = float(matches[2])
        return [Event(step, name, value)]


class PyramidLevelParser:
    pattern = re.compile(
        "\s+Pyramid level\s+(?P<level>\d+)\s+\((?P<width>\d+)\s+x\s+(?P<height>\d+)\)\s+"
    )

    def parse(self, line):
        match = self.pattern.match(line)
        if match is None:
            return None

        step = None
        names = ("level", "width", "height")
        matches = [match.group(group) for group in names]
        return [
            Event(step, name, value) for name, value in zip(names, map(int, matches))
        ]


class ImageOptimizerProgressRecorder(io.StringIO):
    def __init__(self, *args, quiet=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.quiet = quiet
        self._parsers = (OperatorScoreParser(), PyramidLevelParser())
        self._events = []

    def write(self, s):
        self.parse(s)
        if not self.quiet:
            sys.__stdout__.write(s)

    def parse(self, s):
        for parser in self._parsers:
            events = parser.parse(s)
            if events is not None:
                self._events.extend(events)
                break

    def extract(self, filler=np.nan, dropna=False):
        step_offset = 0
        step = 0
        data = defaultdict(list)
        for event in self._events:
            if event.name == "level":
                step_offset = step
                step = 0
            if event.step is not None:
                step = event.step

            data[event.name].append((step + step_offset, event.value))

        num_rows = max([entry[-1][0] for entry in data.values()]) + 1

        def fill(ind, v, **kwargs):
            # FIXME: rename a
            a = np.full((num_rows,), filler)
            np.put(a, ind, v, **kwargs)
            return a

        data = {name: fill(*zip(*entries)) for name, entries in data.items()}
        data["step"] = range(num_rows)
        df = pd.DataFrame.from_dict(data).set_index("step")
        if dropna:
            df = df.dropna(axis="index", how="all")
        return df


@contextmanager
def record_nst(quiet=False):
    recorder = ImageOptimizerProgressRecorder(quiet=quiet)
    with redirect_stdout(recorder):
        yield recorder
