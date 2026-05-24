import csv
import datetime
import json
import os
import tempfile
from collections import defaultdict


class _HumanWriter:
    def writekvs(self, kvs):
        if not kvs:
            return
        key_width = max(len(str(k)) for k in kvs)
        val_width = max(len(f"{float(v):.6g}" if hasattr(v, "__float__") else str(v)) for v in kvs.values())
        line = "-" * (key_width + val_width + 7)
        print(line)
        for key, value in sorted(kvs.items()):
            value = f"{float(value):.6g}" if hasattr(value, "__float__") else str(value)
            print(f"| {str(key):<{key_width}} | {value:<{val_width}} |")
        print(line)

    def writeseq(self, seq):
        print(" ".join(map(str, seq)))

    def close(self):
        pass


class _FileSeqWriter:
    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")

    def writeseq(self, seq):
        self.file.write(" ".join(map(str, seq)) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class _CSVWriter:
    def __init__(self, path):
        self.path = path
        self.keys = []
        self.rows = []

    def writekvs(self, kvs):
        row = dict(kvs)
        self.rows.append(row)
        for key in row:
            if key not in self.keys:
                self.keys.append(key)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def close(self):
        pass


class _JSONWriter:
    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")

    def writekvs(self, kvs):
        clean = {}
        for key, value in kvs.items():
            try:
                clean[key] = float(value)
            except Exception:
                clean[key] = str(value)
        self.file.write(json.dumps(clean) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class Logger:
    CURRENT = None

    def __init__(self, log_dir, output_formats):
        self.dir = log_dir
        self.output_formats = output_formats
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)

    def logkv(self, key, value):
        self.name2val[key] = value

    def logkv_mean(self, key, value):
        old = self.name2val[key]
        cnt = self.name2cnt[key]
        self.name2val[key] = old * cnt / (cnt + 1) + value / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        kvs = dict(self.name2val)
        for writer in self.output_formats:
            if hasattr(writer, "writekvs"):
                writer.writekvs(kvs)
        self.name2val.clear()
        self.name2cnt.clear()
        return kvs

    def log(self, *args):
        for writer in self.output_formats:
            if hasattr(writer, "writeseq"):
                writer.writeseq(args)

    def close(self):
        for writer in self.output_formats:
            writer.close()


def configure(dir=None, format_strs=None, **_):
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = os.path.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("rddm-%Y-%m-%d-%H-%M-%S-%f"),
        )
    os.makedirs(dir, exist_ok=True)
    if format_strs is None:
        format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")

    outputs = []
    for fmt in filter(None, format_strs):
        if fmt == "stdout":
            outputs.append(_HumanWriter())
        elif fmt == "log":
            outputs.append(_FileSeqWriter(os.path.join(dir, "log.txt")))
        elif fmt == "csv":
            outputs.append(_CSVWriter(os.path.join(dir, "progress.csv")))
        elif fmt == "json":
            outputs.append(_JSONWriter(os.path.join(dir, "progress.json")))
        else:
            raise ValueError(f"Unknown log format: {fmt}")
    Logger.CURRENT = Logger(dir, outputs)
    log(f"Logging to {dir}")


def get_current():
    if Logger.CURRENT is None:
        configure()
    return Logger.CURRENT


def logkv(key, value):
    get_current().logkv(key, value)


def logkv_mean(key, value):
    get_current().logkv_mean(key, value)


def dumpkvs():
    return get_current().dumpkvs()


def log(*args, **_):
    get_current().log(*args)
