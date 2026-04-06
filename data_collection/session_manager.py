from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List

from data_collection.config import DataCollectionConfig
from data_collection.models import CollectionSession, EpisodeRecord, TaskSpec, utc_now
from data_collection.raw_storage import (
    ensure_dir,
    episode_metadata_path,
    read_json,
    session_metadata_path,
    write_json,
)


class SessionManager:
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.config.ensure_directories()

    def create_session(
        self,
        benchmark_name: str | None = None,
        device: str | None = None,
        camera: str | None = None,
        target_per_task: int | None = None,
        operator: str = "",
        notes: str = "",
    ) -> CollectionSession:
        now = utc_now()
        session_id = uuid.uuid4().hex[:12]
        benchmark_name = benchmark_name or self.config.benchmark_name
        session_dir = ensure_dir(self.config.runs_dir / session_id)
        ensure_dir(session_dir / "episodes")
        ensure_dir(session_dir / "packaged")
        ensure_dir(session_dir / "validation")
        ensure_dir(session_dir / "exports")
        session = CollectionSession(
            session_id=session_id,
            benchmark_name=benchmark_name,
            created_at=now,
            updated_at=now,
            root_dir=str(session_dir),
            device=device or self.config.default_device,
            camera=camera or self.config.camera,
            target_per_task=target_per_task or self.config.target_per_task,
            operator=operator,
            notes=notes,
            tasks={},
        )
        write_json(session_metadata_path(session_dir), session.to_dict())
        return session

    def load_session(self, session_id: str) -> CollectionSession:
        session_dir = self.config.runs_dir / session_id
        payload = read_json(session_metadata_path(session_dir))
        return CollectionSession(**payload)

    def save_session(self, session: CollectionSession) -> None:
        session.updated_at = utc_now()
        write_json(session_metadata_path(Path(session.root_dir)), session.to_dict())

    def list_sessions(self) -> List[CollectionSession]:
        sessions: List[CollectionSession] = []
        for child in sorted(self.config.runs_dir.iterdir()):
            metadata = session_metadata_path(child)
            if metadata.exists():
                sessions.append(CollectionSession(**read_json(metadata)))
        return sessions

    def create_episode_record(
        self,
        session: CollectionSession,
        task: TaskSpec,
        notes: str = "",
        quality: str = "",
        failure_reason: str = "",
    ) -> EpisodeRecord:
        now = utc_now()
        episode_id = uuid.uuid4().hex[:12]
        episode_dir = ensure_dir(Path(session.root_dir) / "episodes" / episode_id)
        ensure_dir(episode_dir / "raw")
        episode = EpisodeRecord(
            episode_id=episode_id,
            session_id=session.session_id,
            task_name=task.task_name,
            benchmark_name=task.benchmark_name,
            language_instruction=task.language_instruction,
            state="recording",
            created_at=now,
            updated_at=now,
            device=session.device,
            camera=session.camera,
            notes=notes,
            quality=quality,
            failure_reason=failure_reason,
            metadata={"packaged_dataset_path": task.packaged_dataset_path},
        )
        write_json(episode_metadata_path(episode_dir), episode.to_dict())
        return episode

    def save_episode_record(self, session: CollectionSession, episode: EpisodeRecord) -> None:
        episode.updated_at = utc_now()
        episode_dir = Path(session.root_dir) / "episodes" / episode.episode_id
        write_json(episode_metadata_path(episode_dir), episode.to_dict())

    def load_episode_records(self, session: CollectionSession) -> List[EpisodeRecord]:
        episodes_dir = Path(session.root_dir) / "episodes"
        records: List[EpisodeRecord] = []
        if not episodes_dir.exists():
            return records
        for child in sorted(episodes_dir.iterdir()):
            metadata = episode_metadata_path(child)
            if metadata.exists():
                records.append(EpisodeRecord(**read_json(metadata)))
        return records

    def update_task_progress(self, session: CollectionSession, task_name: str, delta: int = 1) -> None:
        session.tasks[task_name] = session.tasks.get(task_name, 0) + delta
        self.save_session(session)

    def summarize_progress(self, session: CollectionSession) -> Dict[str, int]:
        return dict(session.tasks)
