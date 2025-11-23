"""
Utility script to convert all text files to UTF-8 encoding.
Detects the current encoding and converts to UTF-8, overwriting the original file.
"""

from pathlib import Path
from loguru import logger


class EncodingConverter:
    """Service for converting file encodings to UTF-8."""

    ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252", "utf-16", "utf-16le", "utf-16be", "iso-8859-1"]

    @staticmethod
    def detect_and_convert_file(file_path: Path) -> bool:
        """
        Detect file encoding and convert to UTF-8.

        Args:
            file_path: Path to file to convert

        Returns:
            True if converted successfully, False otherwise
        """
        if not file_path.is_file():
            logger.warning(f"Not a file: {file_path}")
            return False

        content = None
        detected_encoding = None

        # First, read as binary
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Check if file was corrupted by previous bad conversion
        # (UTF-16 content incorrectly stored with UTF-8 encoding)
        if raw_data.startswith(b'\xc3\xbf\xc3\xbe') and b'\x00' in raw_data[:100]:
            logger.info(f"Detected corrupted UTF-16 file (bad UTF-8 conversion): {file_path.name}")
            # The file contains UTF-8 encoded bytes that represent UTF-16 LE content
            # We need to reconstruct the original UTF-16 bytes
            try:
                # Read as UTF-8 to get the "text" (which contains the wrong characters)
                corrupted_text = raw_data.decode('utf-8', errors='replace')
                # Re-encode to latin-1 to get back the original byte values
                original_utf16_bytes = corrupted_text.encode('latin-1', errors='replace')
                # Now decode as UTF-16
                content = original_utf16_bytes.decode('utf-16')
                detected_encoding = 'utf-16 (recovered)'
            except Exception as e:
                logger.error(f"Failed to recover corrupted UTF-16 file {file_path.name}: {e}")
                return False
        # Check for real UTF-16 BOM markers
        elif raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
            # UTF-16 BOM detected
            try:
                content = raw_data.decode('utf-16')
                detected_encoding = 'utf-16'
            except Exception as e:
                logger.error(f"Error decoding UTF-16 for {file_path.name}: {e}")
                return False
        elif raw_data.startswith(b'\xef\xbb\xbf'):
            # UTF-8 BOM detected
            try:
                content = raw_data.decode('utf-8-sig')
                detected_encoding = 'utf-8-bom'
            except Exception as e:
                logger.error(f"Error decoding UTF-8 BOM for {file_path.name}: {e}")
                return False
        else:
            # Try to read with different encodings
            for encoding in EncodingConverter.ENCODINGS_TO_TRY:
                try:
                    content = raw_data.decode(encoding)
                    detected_encoding = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    logger.error(f"Error reading {file_path.name} with {encoding}: {e}")
                    return False

        if content is None:
            logger.error(f"Failed to decode {file_path} with any encoding")
            return False

        # Check if already plain UTF-8 without BOM and no null bytes
        if detected_encoding == "utf-8":
            try:
                # Verify it's truly clean UTF-8
                test_encode = content.encode('utf-8')
                if test_encode == raw_data and b'\x00' not in raw_data:
                    logger.debug(f"Already clean UTF-8: {file_path.name}")
                    return True
            except Exception:
                pass

        # Write back as UTF-8 without BOM
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Converted {file_path.name}: {detected_encoding} -> UTF-8")
            return True
        except Exception as e:
            logger.error(f"Error writing {file_path.name}: {e}")
            return False

    @staticmethod
    def convert_directory(directory: Path, pattern: str = "*.txt", recursive: bool = True) -> dict:
        """
        Convert all matching files in directory to UTF-8.

        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively

        Returns:
            Dictionary with conversion statistics
        """
        if not directory.is_dir():
            logger.error(f"Not a directory: {directory}")
            return {"error": "Invalid directory"}

        # Find all matching files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        logger.info(f"Found {len(files)} files matching {pattern}")

        converted = 0
        skipped = 0
        failed = 0

        for file_path in files:
            result = EncodingConverter.detect_and_convert_file(file_path)
            if result:
                converted += 1
            else:
                failed += 1

        stats = {
            "total_files": len(files),
            "converted": converted,
            "failed": failed,
        }

        logger.info(f"Conversion complete: {stats}")
        return stats


def main():
    """Main function to convert files."""
    import sys

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    logger.info("Starting encoding conversion")

    # Convert all txt files in data directory
    data_dir = project_root / "data" / "raw"

    converter = EncodingConverter()
    stats = converter.convert_directory(
        directory=data_dir,
        pattern="*.txt",
        recursive=True
    )

    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
