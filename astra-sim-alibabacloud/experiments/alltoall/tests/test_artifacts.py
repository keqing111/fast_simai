import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simai_a2a.artifacts import (
    read_matrix_artifact,
    read_schedule_artifact,
    write_matrix_artifact,
    write_schedule_artifact,
)
from simai_a2a.model import Schedule, ScheduledFlow, TrafficMatrix


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        values = np.array([[0, 4], [7, 0]], dtype=np.int64)
        self.matrix = TrafficMatrix(values, {"generator": "fixture", "seed": 3})
        self.schedule = Schedule(
            "direct-concurrent",
            [
                ScheduledFlow(0, 0, 1, 4, 0, "direct", 0),
                ScheduledFlow(1, 1, 0, 7, 0, "direct", 0),
            ],
            9,
            {"logical_bytes": 11, "physical_bytes": 11},
        )

    def test_matrix_round_trip_preserves_dtype_metadata_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_matrix_artifact(self.matrix, Path(tmp))
            loaded, loaded_manifest = read_matrix_artifact(Path(tmp))

            self.assertEqual(loaded.values.dtype, np.dtype(np.int64))
            np.testing.assert_array_equal(loaded.values, self.matrix.values)
            self.assertEqual(loaded.metadata, self.matrix.metadata)
            self.assertEqual(loaded_manifest["matrix_sha256"], manifest["matrix_sha256"])
            self.assertEqual(loaded_manifest["logical_bytes"], 11)

    def test_schedule_round_trip_uses_four_column_csv_and_flow_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_schedule_artifact(
                self.schedule,
                root,
                matrix_sha256="a" * 64,
                hardware_sha256="b" * 64,
            )

            with (root / "schedule.csv").open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows, [["0", "1", "4", "0"], ["1", "0", "7", "0"]])
            with (root / "schedule_metadata.csv").open(newline="") as stream:
                metadata_rows = list(csv.DictReader(stream))
            self.assertEqual(
                metadata_rows,
                [
                    {"flow_id": "0", "phase": "direct", "stage_id": "0"},
                    {"flow_id": "1", "phase": "direct", "stage_id": "0"},
                ],
            )

            loaded, loaded_manifest = read_schedule_artifact(root)
            self.assertEqual(loaded, self.schedule)
            self.assertEqual(loaded_manifest["schedule_sha256"], manifest["schedule_sha256"])

    def test_tampered_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_matrix_artifact(self.matrix, root)
            np.save(root / "matrix.npy", np.zeros((3, 3), dtype=np.int64))

            with self.assertRaisesRegex(ValueError, "hash"):
                read_matrix_artifact(root)

    def test_tampered_schedule_or_metadata_is_rejected(self):
        for filename, replacement in (
            ("schedule.csv", "0,1,5,0\n1,0,7,0\n"),
            ("schedule_metadata.csv", "flow_id,phase,stage_id\n0,bad,0\n1,direct,0\n"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_schedule_artifact(
                    self.schedule,
                    root,
                    matrix_sha256="a" * 64,
                    hardware_sha256="b" * 64,
                )
                (root / filename).write_text(replacement, encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "hash"):
                    read_schedule_artifact(root)


if __name__ == "__main__":
    unittest.main()
