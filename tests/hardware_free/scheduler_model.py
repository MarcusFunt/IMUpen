"""Small deterministic discrete-event oracle for FIFO/task/host interleavings.

This is an executable scheduling model, not a timing emulator or proof of MCU
behavior. FIFO word ordering and reset/queue invariants are checked at every
event; all random choices are derived from the stored seed.
"""
from collections import deque
import heapq
import random


class SchedulerModel:
    WATERMARK_WORDS = 48
    CHANNELS = ("gx", "gy", "gz", "ax", "ay", "az")

    def __init__(self, seed, *, frames=96, capacity_words=2048, odr_hz=208.0,
                 max_irq_latency_us=2500, output_queue_limit=64,
                 host_period_us=5000, pause=None, pattern_offset=0,
                 inject_pattern_mismatch=False):
        self.seed = seed
        self.random = random.Random(seed)
        self.params = {"frames": frames, "capacity_words": capacity_words,
                       "odr_hz": odr_hz, "max_irq_latency_us": max_irq_latency_us,
                       "output_queue_limit": output_queue_limit,
                       "host_period_us": host_period_us, "pause": pause,
                       "pattern_offset": pattern_offset,
                       "inject_pattern_mismatch": inject_pattern_mismatch}
        self.period = 1_000_000.0 / odr_hz
        self.clock_us = 0.0
        self.events = []
        self.event_order = 0
        self.trace = []
        self.fifo = deque()
        self.pattern = pattern_offset % 6
        self.overrun = False
        self.irq_level = False
        self.irq_count = 0
        self.pending_edge_time = -1.0
        self.last_used_irq = 0
        self.service_scheduled = False
        self.output_queue = deque()
        self.sent = []
        self.queue_drops = 0
        self.next_seq = 0
        self.epoch = 0
        self.recoveries = 0
        self.batch = 0
        self.decoder_expected = None
        self.assembling = False
        self.partial = [None] * 6
        self.discard_first = True
        self.anchor_irq_times = []
        self.source_frames = {}
        self.initial_phase_pending = pattern_offset % 6
        self.inject_pattern_mismatch = inject_pattern_mismatch

    def schedule(self, at_us, kind):
        self.event_order += 1
        heapq.heappush(self.events, (at_us, self.event_order, kind))

    @staticmethod
    def axes(frame_id):
        return (frame_id, 1000 + frame_id, -1000 - frame_id,
                2000 + frame_id, -2000 - frame_id, 15000 + frame_id)

    def set_level(self, at_us):
        next_level = len(self.fifo) >= self.WATERMARK_WORDS
        rising = next_level and not self.irq_level
        self.irq_level = next_level
        if rising:
            self.irq_count += 1
            self.pending_edge_time = at_us
            latency = self.random.uniform(0.0, self.params["max_irq_latency_us"])
            self.schedule(at_us + latency, "service")
            self.service_scheduled = True
            self.trace.append((round(at_us, 2), "irq_rising", self.irq_count))

    def sensor_sample(self, at_us, frame_id):
        axes = self.axes(frame_id)
        self.source_frames[frame_id] = axes
        start = self.initial_phase_pending
        self.initial_phase_pending = 0
        for value in axes[start:]:
            if len(self.fifo) >= self.params["capacity_words"]:
                self.fifo.popleft()
                self.pattern = (self.pattern + 1) % 6
                self.overrun = True
            self.fifo.append(value)
        self.set_level(at_us)
        self.trace.append((round(at_us, 2), "sample", frame_id, len(self.fifo)))

    def recover_fifo(self, at_us, why):
        self.fifo.clear()
        self.pattern = 0
        self.overrun = False
        self.irq_level = False
        self.decoder_expected = None
        self.assembling = False
        self.partial = [None] * 6
        self.discard_first = True
        self.epoch += 1
        self.recoveries += 1
        self.trace.append((round(at_us, 2), "fifo_reset", self.epoch, why))

    def add_word(self, slot, value, at_us, irq_us, batch):
        if slot == 0:
            self.assembling = True
            self.partial = [None] * 6
        if self.assembling:
            self.partial[slot] = value
        if slot != 5 or not self.assembling:
            return
        self.assembling = False
        frame = tuple(self.partial)
        if self.discard_first:
            self.discard_first = False
            return
        if None in frame:
            raise AssertionError(f"partial axis frame emitted: {frame!r}")
        frame_id = frame[0]
        expected = self.source_frames.get(frame_id)
        if expected != frame:
            raise AssertionError(f"word/channel boundary corruption at frame {frame_id}: {frame!r}")
        sequence = self.next_seq
        self.next_seq += 1
        record = {"seq": sequence, "epoch": self.epoch, "batch": batch,
                  "irq_us": irq_us, "axes": frame}
        if len(self.output_queue) >= self.params["output_queue_limit"]:
            self.queue_drops += 1
        else:
            self.output_queue.append(record)
        self.trace.append((round(at_us, 2), "decoded", sequence, self.epoch, frame_id))

    def service(self, at_us):
        self.service_scheduled = False
        if not self.irq_level:
            self.trace.append((round(at_us, 2), "service_spurious"))
            return
        if self.overrun:
            self.recover_fifo(at_us, "overrun")
            return
        status_pattern = self.pattern
        if self.inject_pattern_mismatch:
            self.inject_pattern_mismatch = False
            status_pattern = (status_pattern + 1) % 6
        if (self.decoder_expected is not None and
                status_pattern != self.decoder_expected):
            self.recover_fifo(at_us, "pattern_mismatch")
            return
        edge_irq_us = self.pending_edge_time if self.irq_count != self.last_used_irq else -1
        if edge_irq_us >= 0:
            self.last_used_irq = self.irq_count
        batch = self.batch
        self.batch += 1
        start_pattern = self.pattern
        for index in range(self.WATERMARK_WORDS):
            if not self.fifo:
                raise AssertionError("watermark status exceeded available FIFO words")
            value = self.fifo.popleft()
            slot = (start_pattern + index) % 6
            self.add_word(slot, value, at_us, edge_irq_us, batch)
            self.pattern = (self.pattern + 1) % 6
        self.decoder_expected = self.pattern
        self.set_level(at_us)
        self.trace.append((round(at_us, 2), "fifo_drain", batch, edge_irq_us, len(self.fifo)))
        if self.irq_level and not self.service_scheduled:
            self.schedule(at_us + self.random.uniform(0.0, 250.0), "service")
            self.service_scheduled = True

    def host_read(self, at_us):
        pause = self.params["pause"]
        paused = pause is not None and pause[0] <= at_us < pause[1]
        if self.output_queue and not paused:
            self.sent.append(self.output_queue.popleft())
        self.trace.append((round(at_us, 2), "host_poll", int(paused), len(self.output_queue)))
        if at_us < self.params["frames"] * self.period + 250_000:
            self.schedule(at_us + self.params["host_period_us"], "host")

    def check_invariants(self):
        if len(self.fifo) > self.params["capacity_words"]:
            raise AssertionError("FIFO exceeded configured capacity")
        if not 0 <= self.pattern < 6:
            raise AssertionError("FIFO pattern escaped [0, 5]")
        sequences = [record["seq"] for record in self.sent]
        if sequences != sorted(set(sequences)):
            raise AssertionError("accepted host sequence is duplicated or reordered")
        if any(a["epoch"] > b["epoch"] for a, b in zip(self.sent, self.sent[1:])):
            raise AssertionError("epoch moved backward in accepted output")
        if self.next_seq != len(self.sent) + self.queue_drops + len(self.output_queue):
            raise AssertionError("assigned sequences are not accounted for by sent, queued, and dropped output")
        for record in self.sent:
            if self.source_frames.get(record["axes"][0]) != record["axes"]:
                raise AssertionError("accepted frame differs from independent source truth")

    def run(self):
        for frame_id in range(self.params["frames"]):
            self.schedule(frame_id * self.period, "sample")
        self.schedule(0.0, "host")
        sample_remaining = self.params["frames"]
        while self.events:
            at_us, _order, kind = heapq.heappop(self.events)
            self.clock_us = at_us
            if kind == "sample":
                frame_id = self.params["frames"] - sample_remaining
                sample_remaining -= 1
                self.sensor_sample(at_us, frame_id)
            elif kind == "service":
                self.service(at_us)
            elif kind == "host":
                self.host_read(at_us)
            self.check_invariants()
        self.check_invariants()
        return {"seed": self.seed, "parameters": self.params,
                "source_frames": len(self.source_frames), "accepted_frames": len(self.sent),
                "queue_drops": self.queue_drops, "fifo_recoveries": self.recoveries,
                "final_epoch": self.epoch, "final_fifo_words": len(self.fifo),
                "anchor_services": sum(record["irq_us"] >= 0 for record in self.sent),
                "sent_sequences": [record["seq"] for record in self.sent],
                "limitations": ["deterministic model; no real FreeRTOS, USB, SPI electrical timing, or analog noise"]}


def make_model(seed, frames=96):
    rnd = random.Random(seed ^ 0x5EED)
    profile = {
        "frames": frames,
        "capacity_words": rnd.choice((96, 144, 2048)),
        "odr_hz": 208.0 * (1.0 + rnd.randint(-500, 500) * 1e-6),
        "max_irq_latency_us": rnd.choice((0.0, 800.0, 6000.0, 80000.0)),
        "output_queue_limit": rnd.choice((4, 16, 64)),
        "host_period_us": rnd.choice((3000, 5000, 12000)),
        "pattern_offset": rnd.randrange(6),
        "inject_pattern_mismatch": rnd.random() < 0.01,
    }
    if rnd.random() < 0.15:
        pause_start = rnd.uniform(40_000, frames * 1_000_000 / 208.0 * 0.65)
        profile["pause"] = (pause_start, pause_start + rnd.uniform(15_000, 120_000))
    else:
        profile["pause"] = None
    return SchedulerModel(seed, **profile)


def run_seed(seed, frames=96):
    return make_model(seed, frames).run()
