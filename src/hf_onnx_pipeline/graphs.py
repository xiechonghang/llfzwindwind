from __future__ import annotations

from pathlib import Path, PurePosixPath
import posixpath

from .common import safe_path


def external_files(graph: Path, repository_root: Path) -> list[str]:
    """Read references without loading tensor payloads; include nested subgraphs."""
    import onnx

    model = onnx.load(str(graph), load_external_data=False)
    found = set()
    graph_relative = graph.relative_to(repository_root).as_posix()

    def walk(message):
        if isinstance(message, onnx.TensorProto):
            if message.data_location == onnx.TensorProto.EXTERNAL:
                data = {v.key: v.value for v in message.external_data}
                location = data.get("location", "")
                if not location or PurePosixPath(location).is_absolute() or "\\" in location:
                    raise ValueError(f"Invalid external data location in {graph.name}")
                relative = posixpath.normpath(posixpath.join(posixpath.dirname(graph_relative), location))
                safe_path(repository_root, relative)
                found.add(relative)
        for field, value in message.ListFields():
            if field.message_type is not None:
                if field.is_repeated:
                    for child in value:
                        walk(child)
                else:
                    walk(value)

    walk(model)
    return sorted(found)


def inspect_graphs(root: Path, graphs: list[str], runtime: bool = False) -> dict:
    import onnx

    if not graphs:
        raise ValueError("No ONNX graphs selected")
    results = []
    for name in graphs:
        path = safe_path(root, name)
        dependencies = external_files(path, root)
        for item in dependencies:
            if not safe_path(root, item).is_file():
                raise ValueError(f"Missing external tensor data: {item}")
        # Path API avoids protobuf's in-memory size limit for external-data models.
        onnx.checker.check_model(str(path))
        proto = onnx.load(str(path), load_external_data=False)
        record = {"file": name, "external_files": dependencies,
                  "inputs": [x.name for x in proto.graph.input],
                  "outputs": [x.name for x in proto.graph.output],
                  "opsets": [{"domain": x.domain, "version": x.version} for x in proto.opset_import],
                  "structure": "passed", "runtime_load": "not_run"}
        if runtime:
            import onnxruntime as ort
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            record["runtime_load"] = "passed"
            del session
        results.append(record)
    return {"graphs": results, "numerical_validation": "not_run"}
