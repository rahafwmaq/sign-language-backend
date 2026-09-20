from pathlib import Path
import shutil
import zipfile

from fastapi import UploadFile

from shared.file_manager import FileManager


PathLike = str | Path


class ZipManager:
    @staticmethod
    def save_upload_file(
        upload_file: UploadFile,
        destination: PathLike,
    ) -> Path:
        destination = Path(destination)

        FileManager.ensure_dir(destination.parent)

        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return destination

    @staticmethod
    def extract(
        zip_path: PathLike,
        destination: PathLike,
        *,
        clean_before_extract: bool = True,
    ) -> Path:
        zip_path = Path(zip_path)
        destination = Path(destination)

        if not zip_path.exists():
            raise FileNotFoundError(...)

        if not zipfile.is_zipfile(zip_path):
            raise ValueError(
                "Invalid ZIP file."
            )

        if clean_before_extract:
            FileManager.delete_dir(destination)

        FileManager.ensure_dir(destination)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(destination)

        return destination

    @staticmethod
    def compress(
        source_directory: PathLike,
        output_zip: PathLike,
    ) -> Path:
        source_directory = Path(source_directory)
        output_zip = Path(output_zip)

        FileManager.ensure_dir(output_zip.parent)

        with zipfile.ZipFile(
            output_zip,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zip_file:
            for file in source_directory.rglob("*"):
                if file.is_file():
                    zip_file.write(
                        file,
                        file.relative_to(source_directory),
                    )

        return output_zip