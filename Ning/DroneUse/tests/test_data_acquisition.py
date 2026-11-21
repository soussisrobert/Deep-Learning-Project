"""No-network deterministic fixtures for bounded authoritative acquisition."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import stat
import zipfile

import numpy as np
from PIL import Image
import pytest

from hift_research.data_acquisition import AcquisitionError, HTTPRangeReader
from hift_research.data_acquisition.uav import (
    extract_member, parse_sequence_config, safe_member_target, validate_sequence,
)


class FakeResponse(io.BytesIO):
    def __init__(self, data, start, end, total, *, status=206, modified="fixed", range_override=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": range_override or f"bytes {start}-{end}/{total}",
                        "Content-Length": str(end - start + 1), "Last-Modified": modified}
        self.read_count = 0

    def read(self, size=-1):
        self.read_count += 1
        return super().read(size)


class FakeHTTP:
    def __init__(self, payload):
        self.payload, self.requests, self.responses = payload, [], []
        self.status, self.modified, self.range_override = 206, "fixed", None
        self.truncate = 0

    def __call__(self, request, timeout):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match
        start, end = map(int, match.groups())
        self.requests.append((start, end))
        content = self.payload[start:end + 1]
        if self.truncate:
            content = content[:-self.truncate]
        response = FakeResponse(content, start, end, len(self.payload), status=self.status,
                                modified=self.modified, range_override=self.range_override)
        self.responses.append(response)
        return response


def reader_for(payload, **kwargs):
    server = FakeHTTP(payload)
    return HTTPRangeReader("https://official.example/archive.zip", len(payload), opener=server, **kwargs), server


def test_exact_ranges_cache_seek_and_budget(tmp_path):
    reader, server = reader_for(bytes(range(100)), budget_bytes=10, cache_dir=tmp_path / "cache")
    reader.seek(20)
    assert reader.read(5) == bytes(range(20, 25))
    assert reader.network_bytes == 5
    reader.seek(21)
    assert reader.read(3) == bytes(range(21, 24))
    assert len(server.requests) == 1
    reader.seek(-2, io.SEEK_END)
    assert reader.read() == bytes([98, 99])
    assert reader.network_bytes == 7
    reader.seek(50)
    with pytest.raises(AcquisitionError, match="budget"):
        reader.read(4)
    assert len(server.requests) == 2  # Oversized request was never issued.
    assert reader.records[0]["sha256"] == hashlib.sha256(bytes(range(20, 25))).hexdigest()
    assert Path(reader.records[0]["cache_path"]).exists()


def test_server_ignoring_range_never_consumed():
    reader, server = reader_for(b"whole huge archive would be here")
    server.status = 200
    with pytest.raises(AcquisitionError, match="ignored Range"):
        reader.read(1)
    assert server.responses[0].read_count == 0
    assert reader.network_bytes == 0


def test_content_range_and_size_change_rejected_before_body_read():
    reader, server = reader_for(b"123456789")
    server.range_override = "bytes 0-1/10"
    with pytest.raises(AcquisitionError, match="Content-Range"):
        reader.read(2)
    assert server.responses[0].read_count == 0


def test_last_modified_change_rejected():
    reader, server = reader_for(b"123456789")
    assert reader.read(2) == b"12"
    server.modified = "changed"
    reader.seek(5)
    with pytest.raises(AcquisitionError, match="changed Last-Modified"):
        reader.read(2)
    assert server.responses[-1].read_count == 0


def test_truncated_response_rejected_and_received_bytes_counted():
    reader, server = reader_for(b"123456789")
    server.truncate = 1
    with pytest.raises(AcquisitionError, match="truncated"):
        reader.read(3)
    assert reader.network_bytes == 2


def test_prefetch_chunk_boundary_no_extra_network_and_zip_access():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("official/configSeqs.m", "original bytes")
        archive.writestr("official/sequence/000001.jpg", b"not decoded by ZIP test")
    data = payload.getvalue()
    reader, server = reader_for(data)
    reader.prefetch(0, len(data) - 1, chunk_bytes=40)
    before = reader.network_bytes
    with zipfile.ZipFile(reader) as archive:
        assert archive.read("official/configSeqs.m") == b"original bytes"
    assert reader.network_bytes == before == len(data)
    assert all(end - start + 1 <= 40 for start, end in server.requests)


@pytest.mark.parametrize("url,size,budget", [("http://official.example/a", 10, 10),
    ("https://official.example/a", 0, 10), ("https://official.example/a", 10, 0)])
def test_unsafe_reader_parameters(url, size, budget):
    with pytest.raises(AcquisitionError):
        HTTPRangeReader(url, size, budget_bytes=budget)


def test_safe_member_paths_and_no_overwrites(tmp_path):
    for name in ["../escape.jpg", "/absolute.jpg", "C:/file.jpg", "x\\evil.jpg"]:
        with pytest.raises(AcquisitionError):
            safe_member_target(tmp_path, name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("official/annotation.txt", b"1,2,3,4\n")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        info = archive.getinfo("official/annotation.txt")
        record = extract_member(archive, info, tmp_path)
        assert Path(record["path"]).read_bytes() == b"1,2,3,4\n"
        with pytest.raises(AcquisitionError, match="overwrite"):
            extract_member(archive, info, tmp_path)
        with pytest.raises(AcquisitionError, match="oversized"):
            extract_member(archive, info, tmp_path, max_bytes=2)


def test_symlink_and_encrypted_archive_entries_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("annotation.txt", b"original")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        info = archive.getinfo("annotation.txt")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with pytest.raises(AcquisitionError, match="symlink"):
            extract_member(archive, info, tmp_path)
        info.external_attr = 0
        info.flag_bits |= 1
        with pytest.raises(AcquisitionError, match="encrypted"):
            extract_member(archive, info, tmp_path)


def test_official_matlab_config_parsed_never_executed():
    text = """function seqs=configSeqs
    seqs = {struct('name','test_1','path','D:\\data_seq\\test\\','startFrame',2,'endFrame',4,'nz',6,'ext','jpg','init_rect',[0,0,0,0])};
    system('DO NOT EXECUTE');
    """
    result = parse_sequence_config(text)
    assert result == {"test_1": {"folder": "test", "start_frame": 2, "end_frame": 4, "digits": 6, "extension": "jpg"}}
    with pytest.raises(AcquisitionError, match="duplicate"):
        parse_sequence_config(text + text)
    with pytest.raises(AcquisitionError):
        parse_sequence_config(text.replace("'endFrame',4", "'endFrame',1"))
    with pytest.raises(AcquisitionError):
        parse_sequence_config("not an official sequence config")


def real_jpeg_fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for frame_id in range(1, 4):
        # Generated imagery is ONLY for unit tests, never written to acquired data paths.
        Image.fromarray(np.full((12, 16, 3), frame_id, dtype=np.uint8)).save(images / f"{frame_id:06d}.jpg")
    annotation = tmp_path / "annotation.txt"
    annotation.write_bytes(b"1,2,3,4\nnan,nan,nan,nan\n2,3,4,5\n")
    config = {"start_frame": 1, "end_frame": 3, "digits": 6, "extension": "jpg"}
    return images, annotation, config


def test_complete_sequence_decoding_and_absence_retained(tmp_path):
    images, annotation, config = real_jpeg_fixture(tmp_path)
    original_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
    result = validate_sequence(images, annotation, config)
    assert result["frame_count"] == 3
    assert result["all_frames_decoded"]
    assert result["absent_frame_ids"] == [2]
    assert result["annotation_rows_dropped"] == 0
    assert result["annotation_sha256"] == original_hash
    assert result["frames"][0]["width"] == 16
    assert result["initial_box_annotation_native"] == [1, 2, 3, 4]


def test_missing_or_extra_frame_id_and_corruption_rejected(tmp_path):
    images, annotation, config = real_jpeg_fixture(tmp_path)
    frame = images / "000002.jpg"
    frame.rename(images / "000004.jpg")
    with pytest.raises(AcquisitionError, match="ID set mismatch"):
        validate_sequence(images, annotation, config)
    (images / "000004.jpg").rename(frame)
    frame.write_bytes(b"not a JPEG")
    with pytest.raises(AcquisitionError, match="decode failed"):
        validate_sequence(images, annotation, config)


@pytest.mark.parametrize("row", ["nan,2,3,4", "1,2,0,4", "1,2,-3,4", "inf,2,3,4"])
def test_malformed_boxes_are_not_reinterpreted_as_absence(tmp_path, row):
    images, annotation, config = real_jpeg_fixture(tmp_path)
    annotation.write_text("1,2,3,4\n" + row + "\n2,3,4,5\n", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="malformed"):
        validate_sequence(images, annotation, config)


def test_missing_annotation_row_never_truncates_images(tmp_path):
    images, annotation, config = real_jpeg_fixture(tmp_path)
    annotation.write_text("1,2,3,4\n" * 2, encoding="utf-8")
    with pytest.raises(AcquisitionError, match="count mismatch"):
        validate_sequence(images, annotation, config)
