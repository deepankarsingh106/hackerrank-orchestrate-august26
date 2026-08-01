"""Centralized configuration for the Message Notification Router pipeline."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CODE_DIR_NAME: Final[str] = "code"
DATASET_DIR_NAME: Final[str] = "dataset"
MEDIA_DIR_NAME: Final[str] = "media"
IMAGES_SUBDIR: Final[str] = "images"
AUDIO_SUBDIR: Final[str] = "audio"
LOGS_DIR_NAME: Final[str] = "logs"
OUTPUT_DIR_NAME: Final[str] = "output"
DEFAULT_OUTPUT_FILENAME: Final[str] = "output.csv"
MESSAGES_FILENAME: Final[str] = "messages.csv"
SAMPLE_MESSAGES_FILENAME: Final[str] = "sample_messages.csv"
EVIDENCE_SEPARATOR: Final[str] = ";"
EVIDENCE_NONE: Final[str] = "none"
ENV_FILE_NAME: Final[str] = ".env"


class Action(StrEnum):
    """Allowed routing actions for incoming messages."""

    NOTIFY = "notify"
    DIGEST = "digest"
    MUTE = "mute"


class MessageType(StrEnum):
    """Allowed message classification labels."""

    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Confidence cutoffs used by rule engine and calibrator stages."""

    high: float = 0.85
    medium: float = 0.65
    low: float = 0.40
    rule_engine_auto: float = 0.80
    gemini_minimum: float = 0.50
    notify_minimum: float = 0.70
    digest_minimum: float = 0.45

    def __post_init__(self) -> None:
        for name, value in (
            ("high", self.high),
            ("medium", self.medium),
            ("low", self.low),
            ("rule_engine_auto", self.rule_engine_auto),
            ("gemini_minimum", self.gemini_minimum),
            ("notify_minimum", self.notify_minimum),
            ("digest_minimum", self.digest_minimum),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Confidence threshold '{name}' must be between 0 and 1, got {value}."
                )


@dataclass(frozen=True)
class GeminiConfig:
    """Settings for the Gemini decision engine."""

    model_name: str = "gemini-2.0-flash"
    api_key: str = ""
    temperature: float = 0.1
    max_output_tokens: int = 1024
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("Gemini model name must not be empty.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"Gemini temperature must be between 0 and 2, got {self.temperature}."
            )
        if self.max_output_tokens <= 0:
            raise ValueError("Gemini max_output_tokens must be positive.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("Gemini request_timeout_seconds must be positive.")


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration for the application."""

    level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "router.log"
    log_format: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used across the pipeline."""

    project_root: Path
    code_dir: Path
    dataset_root: Path
    media_root: Path
    images_dir: Path
    audio_dir: Path
    logs_dir: Path
    output_dir: Path
    output_file: Path
    messages_file: Path
    sample_messages_file: Path
    env_file: Path

    def resolve_media_file(self, relative_path: str | Path) -> Path:
        """Resolve a dataset-relative media path to an absolute file path."""
        relative = Path(relative_path)
        if relative.is_absolute():
            return relative.resolve()
        return (self.dataset_root / relative).resolve()

    def resolve_dataset_file(self, filename: str) -> Path:
        """Resolve a CSV filename inside the dataset root."""
        return (self.dataset_root / filename).resolve()

    def ensure_runtime_directories(self) -> None:
        """Create directories required at runtime if they do not exist."""
        for directory in (self.logs_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    paths: PathConfig
    gemini: GeminiConfig
    confidence: ConfidenceThresholds
    logging: LoggingSettings

    def configure_logging(self) -> None:
        """Apply logging settings to the root logger."""
        configure_logging(self.logging, self.paths.logs_dir)


def resolve_code_dir(start: Path | None = None) -> Path:
    """Return the absolute path to the `code/` directory."""
    if start is None:
        start = Path(__file__).resolve().parent
    else:
        start = start.resolve()

    if start.name == CODE_DIR_NAME:
        return start

    candidate = start / CODE_DIR_NAME
    if candidate.is_dir():
        return candidate.resolve()

    if (start.parent / CODE_DIR_NAME).is_dir():
        return (start.parent / CODE_DIR_NAME).resolve()

    raise FileNotFoundError(
        f"Unable to locate '{CODE_DIR_NAME}' directory from start path: {start}"
    )


def resolve_project_root(start: Path | None = None) -> Path:
    """Return the repository root directory containing `code/` and `dataset/`."""
    code_dir = resolve_code_dir(start)
    project_root = code_dir.parent

    dataset_dir = project_root / DATASET_DIR_NAME
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dataset directory at {dataset_dir}, but it was not found."
        )

    return project_root.resolve()


def resolve_dataset_root(
    project_root: Path | None = None,
    override: str | Path | None = None,
) -> Path:
    """Return the dataset root, optionally overridden by environment or argument."""
    if override is not None:
        dataset_root = Path(override).expanduser().resolve()
    else:
        root = project_root or resolve_project_root()
        dataset_root = (root / DATASET_DIR_NAME).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    return dataset_root


def build_path_config(
    project_root: Path | None = None,
    dataset_override: str | Path | None = None,
    output_override: str | Path | None = None,
) -> PathConfig:
    """Construct path configuration from project layout and optional overrides."""
    root = (project_root or resolve_project_root()).resolve()
    code_dir = root / CODE_DIR_NAME
    dataset_root = resolve_dataset_root(root, dataset_override)

    media_root = dataset_root / MEDIA_DIR_NAME
    images_dir = media_root / IMAGES_SUBDIR
    audio_dir = media_root / AUDIO_SUBDIR
    logs_dir = code_dir / LOGS_DIR_NAME
    output_dir = code_dir / OUTPUT_DIR_NAME

    if output_override is not None:
        output_file = Path(output_override).expanduser().resolve()
    else:
        output_file = (dataset_root / DEFAULT_OUTPUT_FILENAME).resolve()

    return PathConfig(
        project_root=root,
        code_dir=code_dir,
        dataset_root=dataset_root,
        media_root=media_root,
        images_dir=images_dir,
        audio_dir=audio_dir,
        logs_dir=logs_dir,
        output_dir=output_dir,
        output_file=output_file,
        messages_file=(dataset_root / MESSAGES_FILENAME).resolve(),
        sample_messages_file=(dataset_root / SAMPLE_MESSAGES_FILENAME).resolve(),
        env_file=(root / ENV_FILE_NAME).resolve(),
    )


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse common truthy/falsey environment string values."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {value}")


def _parse_float(name: str, value: str | None, default: float) -> float:
    """Parse a float environment variable with validation."""
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float.") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"Environment variable {name} must be between 0 and 1.")
    return parsed


def load_env(project_root: Path | None = None) -> Path:
    """Load environment variables from the project `.env` file if present."""
    root = (project_root or resolve_project_root()).resolve()
    env_file = root / ENV_FILE_NAME

    if env_file.is_file():
        loaded = load_dotenv(dotenv_path=env_file, override=False)
        if loaded:
            logger.debug("Loaded environment variables from %s", env_file)
        else:
            logger.debug("No new environment variables loaded from %s", env_file)
    else:
        logger.debug("Environment file not found at %s; using process environment only")

    return env_file


def configure_logging(settings: LoggingSettings, logs_dir: Path) -> None:
    """Configure root logging handlers according to settings."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    level_name = settings.level.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {settings.level}")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt=settings.date_format,
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    if settings.log_to_file:
        log_file = logs_dir / settings.log_filename
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        logger.info("File logging enabled at %s", log_file)


def load_config(
    project_root: Path | None = None,
    configure_logs: bool = True,
) -> AppConfig:
    """Load application configuration from environment variables and project layout."""
    root = (project_root or resolve_project_root()).resolve()
    load_env(root)

    dataset_override = os.getenv("DATASET_ROOT")
    output_override = os.getenv("OUTPUT_PATH")

    paths = build_path_config(
        project_root=root,
        dataset_override=dataset_override,
        output_override=output_override,
    )
    paths.ensure_runtime_directories()

    gemini = GeminiConfig(
        model_name=os.getenv("GEMINI_MODEL", GeminiConfig.model_name),
        api_key=os.getenv("GEMINI_API_KEY", ""),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")),
        request_timeout_seconds=float(
            os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "60.0")
        ),
    )

    confidence = ConfidenceThresholds(
        high=_parse_float("CONFIDENCE_HIGH", os.getenv("CONFIDENCE_HIGH"), 0.85),
        medium=_parse_float(
            "CONFIDENCE_MEDIUM", os.getenv("CONFIDENCE_MEDIUM"), 0.65
        ),
        low=_parse_float("CONFIDENCE_LOW", os.getenv("CONFIDENCE_LOW"), 0.40),
        rule_engine_auto=_parse_float(
            "CONFIDENCE_RULE_ENGINE_AUTO",
            os.getenv("CONFIDENCE_RULE_ENGINE_AUTO"),
            0.80,
        ),
        gemini_minimum=_parse_float(
            "CONFIDENCE_GEMINI_MINIMUM",
            os.getenv("CONFIDENCE_GEMINI_MINIMUM"),
            0.50,
        ),
        notify_minimum=_parse_float(
            "CONFIDENCE_NOTIFY_MINIMUM",
            os.getenv("CONFIDENCE_NOTIFY_MINIMUM"),
            0.70,
        ),
        digest_minimum=_parse_float(
            "CONFIDENCE_DIGEST_MINIMUM",
            os.getenv("CONFIDENCE_DIGEST_MINIMUM"),
            0.45,
        ),
    )

    logging_settings = LoggingSettings(
        level=os.getenv("LOG_LEVEL", LoggingSettings.level),
        log_to_file=_parse_bool(os.getenv("LOG_TO_FILE"), True),
        log_filename=os.getenv("LOG_FILENAME", LoggingSettings.log_filename),
    )

    config = AppConfig(
        paths=paths,
        gemini=gemini,
        confidence=confidence,
        logging=logging_settings,
    )

    if configure_logs:
        config.configure_logging()

    logger.info("Configuration loaded successfully")
    logger.info("Project root: %s", config.paths.project_root)
    logger.info("Dataset root: %s", config.paths.dataset_root)
    logger.info("Output file: %s", config.paths.output_file)
    logger.info("Gemini model: %s", config.gemini.model_name)

    return config


def validate_required_paths(config: AppConfig) -> None:
    """Validate that critical input paths exist before pipeline execution."""
    required_files = (
        config.paths.messages_file,
        config.paths.sample_messages_file,
    )

    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        formatted = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Required dataset files are missing: {formatted}")

    for directory in (
        config.paths.dataset_root,
        config.paths.media_root,
        config.paths.images_dir,
        config.paths.audio_dir,
    ):
        if not directory.exists():
            logger.warning("Expected directory does not exist: %s", directory)


def main() -> int:
    """Load and display configuration for standalone execution."""
    try:
        config = load_config(configure_logs=True)
        validate_required_paths(config)
    except Exception:
        logger.exception("Failed to load configuration")
        return 1

    print("Message Notification Router configuration")
    print(f"  Project root     : {config.paths.project_root}")
    print(f"  Dataset root     : {config.paths.dataset_root}")
    print(f"  Media root       : {config.paths.media_root}")
    print(f"  Images directory : {config.paths.images_dir}")
    print(f"  Audio directory  : {config.paths.audio_dir}")
    print(f"  Output file      : {config.paths.output_file}")
    print(f"  Logs directory   : {config.paths.logs_dir}")
    print(f"  Gemini model     : {config.gemini.model_name}")
    print(f"  API key present  : {'yes' if config.gemini.api_key else 'no'}")
    print(f"  Confidence high  : {config.confidence.high}")
    print(f"  Allowed actions  : {[action.value for action in Action]}")
    print(
        f"  Allowed types    : {[message_type.value for message_type in MessageType]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
