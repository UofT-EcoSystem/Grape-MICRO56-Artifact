import csv
import json
import logging
import os
import time

import numpy as np

from .format import bold, emph

logger = logging.getLogger("quik_fix")


def set_logging_format():
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d, %(levelname)s] %(message)s",
    )
    logger.setLevel(logging.INFO)


class CSVStatsLogger:
    __slots__ = "fin", "writer", "output_format", "global_attrs"

    def __init__(self, filename, force_overwrite=False, output_format="{:.3e}"):
        file_exists = os.path.exists(filename)
        self.fin = open(filename, "wt" if force_overwrite else "at")
        self.writer = csv.DictWriter(
            self.fin, ["Name", "Attrs", "Avg", "Std", "min", "Median", "MAX"]
        )
        self.output_format = output_format
        if force_overwrite or (not file_exists):
            self.writer.writeheader()
            self.fin.flush()
        self.global_attrs = {}

    def write(self, name, data, attrs=None):
        runtime_attrs = {} if not attrs else attrs
        attrs = {**self.global_attrs, **runtime_attrs}
        if data is None:
            self.writer.writerow(
                {
                    "Name": name,
                    "Attrs": json.dumps(attrs),
                    "Avg": None,
                    "Std": None,
                    "min": None,
                    "Median": None,
                    "MAX": None,
                }
            )
            logger.info(
                emph(name)
                + (f" attrs={attrs}" if attrs is not None else " ")
                + ": (-)+(-)"
            )
        else:
            mean, std, median = np.average(data), np.std(data), np.median(data)
            self.writer.writerow(
                {
                    "Name": name,
                    "Attrs": json.dumps(attrs),
                    "Avg": self.output_format.format(mean),
                    "Std": self.output_format.format(std),
                    "min": self.output_format.format(np.min(data)),
                    "Median": self.output_format.format(median),
                    "MAX": self.output_format.format(np.max(data)),
                }
            )
            logger.info(
                emph(name)
                + (f" attrs={attrs}: " if attrs is not None else " ")
                + bold(self.output_format.format(mean))
                + "+"
                + self.output_format.format(std)
                + " "
                + "(M="
                + self.output_format.format(median)
                + ")"
            )
        self.fin.flush()


class CSVSpeedometer(object):
    """
    Log

    - Training Throughput (Samples per Second),
    - Memory Usage (from `nvidia-smi`)
    - Evaluation Metrics
    - Power and Energy Consumption

    periodically, and at the same time, dump the results using CSV file.
    """

    __slots__ = [
        "num_training_samples",
        "log_frequency",
        "tic",
        "global_step",
        "global_training_samples",
        "enable_gpu_query",
        "gpu_uuid_to_id",
        "gpu_uuid_to_energy",
        "writer",
    ]

    def __init__(
        self,
        log_frequency=1,
        csv_filename="speedometer.csv",
        enable_gpu_query=True,
        eval_metric_names=None,
    ):
        from .nvml import (  # pylint: disable=import-outside-toplevel
            GPUQueryKind,
            query_gpu_status,
        )

        self.num_training_samples = 0
        self.log_frequency = log_frequency
        self.tic = time.perf_counter()
        # `global_step` records the number of training batches. It is
        # incremented every time the `Speedometer` is called.
        self.global_step = 0
        self.global_training_samples = 0
        self.enable_gpu_query = enable_gpu_query

        if self.enable_gpu_query:
            query_result = query_gpu_status(GPUQueryKind.UUID)
            self.gpu_uuid_to_id = {}
            self.gpu_uuid_to_energy = {}
            for i, gpu_uuid in enumerate(query_result):
                self.gpu_uuid_to_id[gpu_uuid] = i
                # initialize energy to an array of zeros
                self.gpu_uuid_to_energy[gpu_uuid] = 0.0

        try:
            # cleanup previous output file, if there exists
            os.remove(csv_filename)
        except OSError:
            pass

        csv_header = (
            [
                "Global Step",
                "Global Training Samples",
                "Speed (samples/s)",
                "Device Index",
                "Memory Usage (MB)",
                "Power (W)",
                "Energy (J)",
            ]
            if self.enable_gpu_query
            else [
                "Global Step",
                "Global Training Samples",
                "Speed (samples/s)",
            ]
        )
        if eval_metric_names is not None:
            if not isinstance(eval_metric_names, list):
                eval_metric_names = [eval_metric_names]
            csv_header.extend(eval_metric_names)
        self.writer = csv.DictWriter(open(csv_filename, "wt"), csv_header)
        self.writer.writeheader()

    def __call__(self, batch_size, eval_metrics=None):
        """
        Callback to show Training Throughput, Memory Usage, Evaluation Metrics
        """
        from .nvml import (  # pylint: disable=import-outside-toplevel
            GPUQueryKind,
            query_gpu_status,
        )

        self.num_training_samples += batch_size
        self.global_step += 1
        self.global_training_samples += batch_size

        if self.global_step % self.log_frequency == 0:
            # Training Throughput
            time_diff = time.perf_counter() - self.tic
            speed = self.num_training_samples / time_diff

            if self.enable_gpu_query:
                gpu_uuid_to_used_memory = {}
                gpu_uuid_to_power = {}

                # NVIDIA Docker containers have trouble in querying process
                # names. Therefore process names are temporarily ignored.
                query_result = query_gpu_status(GPUQueryKind.MEMORY)

                for gpu_uuid, pid, used_gpu_memory in query_result:
                    if pid != os.getpid():
                        logger.warning(f"Skipping pid={pid} != this_pid={os.getpid()}")
                        continue
                    gpu_uuid_to_used_memory[gpu_uuid] = used_gpu_memory

                query_result = query_gpu_status(GPUQueryKind.POWER)

                for gpu_uuid, power in query_result:
                    gpu_uuid_to_power[gpu_uuid] = power
                    self.gpu_uuid_to_energy[gpu_uuid] += power * time_diff

                for gpu_uuid, used_memory in gpu_uuid_to_used_memory.items():
                    log_entry = {
                        "Global Step": self.global_step,
                        "Global Training Samples": self.global_training_samples,
                        "Speed (samples/s)": speed,
                        "Device Index": self.gpu_uuid_to_id[gpu_uuid],
                        "Memory Usage (MB)": used_memory,
                        "Power (W)": gpu_uuid_to_power[gpu_uuid],
                        "Energy (J)": self.gpu_uuid_to_energy[gpu_uuid],
                    }
                    if eval_metrics is not None:
                        log_entry.update(eval_metrics)
                    log_entry_str = ""
                    for k, entry in log_entry.items():
                        log_entry_str += bold(k) + f" : {entry}" "\n"
                    logger.info(f"{log_entry_str}")
                    self.writer.writerow(log_entry)
            else:
                log_entry = {
                    "Global Step": self.global_step,
                    "Global Training Samples": self.global_training_samples,
                    "Speed (samples/s)": speed,
                }
                if eval_metrics is not None:
                    log_entry.update(eval_metrics)
                log_entry_str = ""
                for k, entry in log_entry.items():
                    log_entry_str += bold(k) + f" : {entry}" "\n"
                logger.info(f"{log_entry_str}")
                self.writer.writerow(log_entry)

            self.num_training_samples = 0
            self.tic = time.perf_counter()
