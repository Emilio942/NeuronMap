from pathlib import Path

from src.zoo.artifact_manager import ArtifactManager
from src.zoo.artifact_schema import (
    ArtifactSchema,
    ArtifactSearchFilter,
    ArtifactType,
    AuthorInfo,
    LicenseType,
)


def _build_artifact(index: int) -> ArtifactSchema:
    return ArtifactSchema(
        uuid=f"artifact-{index:03d}",
        name=f"Artifact {index}",
        version="1.0.0",
        artifact_type=ArtifactType.SAE_MODEL,
        description="This is a valid description with more than 10 chars",
        authors=[AuthorInfo(name="Tester")],
        license=LicenseType.MIT,
        model_compatibility=[{"model_name": "gpt2"}],
        downloads=index,
        total_size_bytes=index * 10,
    )


def test_get_statistics_includes_all_metadata_files(tmp_path: Path):
    manager = ArtifactManager(tmp_path / "storage")

    # Create more artifacts than default list limit (50)
    for index in range(55):
        artifact = _build_artifact(index)
        metadata_path = manager.metadata_dir / f"{artifact.uuid}.json"
        metadata_path.write_text(artifact.model_dump_json(indent=2))

    stats = manager.get_statistics()

    assert stats["total_artifacts"] == 55
    assert stats["total_downloads"] == sum(range(55))


def test_list_artifacts_invalid_sort_field_falls_back_instead_of_crashing(tmp_path: Path):
    manager = ArtifactManager(tmp_path / "storage")

    for index in range(2):
        artifact = _build_artifact(index)
        metadata_path = manager.metadata_dir / f"{artifact.uuid}.json"
        metadata_path.write_text(artifact.model_dump_json(indent=2))

    result = manager.list_artifacts(
        filter_params=ArtifactSearchFilter.model_construct(sort_by="does_not_exist")
    )

    assert result.total_count == 2
    assert len(result.artifacts) == 2
