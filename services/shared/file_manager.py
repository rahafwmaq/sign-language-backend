from pathlib import Path
import shutil


PathLike = str | Path


class FileManager:
    @staticmethod
    def ensure_dir(path: PathLike) -> Path:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def delete_dir(path: PathLike) -> bool:
        directory = Path(path)

        if not directory.exists():
            return False

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory}"
            )

        shutil.rmtree(directory)
        return True

    @staticmethod
    def delete_file(path: PathLike) -> bool:
        file_path = Path(path)

        if not file_path.exists():
            return False

        if not file_path.is_file():
            raise IsADirectoryError(
                f"Path is not a file: {file_path}"
            )

        file_path.unlink()
        return True

    @staticmethod
    def list_files(
        path: PathLike,
        extensions: tuple[str, ...],
        *,
        recursive: bool = True,
    ) -> list[Path]:
        directory = Path(path)

        if not directory.exists():
            return []

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory}"
            )

        normalized_extensions = tuple(
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in extensions
        )

        iterator = (
            directory.rglob("*")
            if recursive
            else directory.glob("*")
        )

        return sorted(
            file_path
            for file_path in iterator
            if file_path.is_file()
            and file_path.suffix.lower() in normalized_extensions
        )
    
    @staticmethod
    def copy_file(
        source: PathLike,
        destination: PathLike,
    ) -> Path:
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(
                f"Source file not found: {source}"
            )

        FileManager.ensure_dir(destination.parent)

        shutil.copy2(source, destination)

        return destination
    
    @staticmethod
    def exists(path: PathLike) -> bool:
        return Path(path).exists()