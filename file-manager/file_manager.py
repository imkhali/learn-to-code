"""
File Manager - Pattern Practice Project
Your task: Fill in the TODO sections!

Patterns to implement:
- Observer: File change notifications
- Strategy: Sort and filter files
- Command: File operations with undo
- Factory: Create different file types
- Decorator: Add metadata to files
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from enum import Enum
import os


class FileEvent(Enum):
    FILE_ADDED = "File added"
    FILE_DELETED = "File deleted"
    FILE_MODIFIED = "File modified"
    FILE_COPIED = "File copied"
    FILE_MOVED = "File moved"


# ============================================================================
# OBSERVER PATTERN - TODO: Implement this!
# ============================================================================


class Observer(ABC):
    """Base observer class"""

    @abstractmethod
    def update(self, event: FileEvent, data: dict):
        pass


class Subject:
    """Base subject class for observables"""

    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event: FileEvent, data: dict):
        for observer in self._observers:
            observer.update(event, data)


class FileChangeObserver(Observer):
    """Observer that prints file change notifications"""

    def update(self, event: FileEvent, data: dict):

        timestamp = datetime.now().strftime("%H:%M:%S")
        filename = data.get("filename", "Unknown")

        # Event-specific formatting with extra details
        match event:
            case FileEvent.FILE_ADDED:
                size = data.get("size", "?")
                message = f"[{timestamp}] ✅ File added: {filename} ({size} bytes)"
            case FileEvent.FILE_DELETED:
                message = f"[{timestamp}] ❌ File deleted: {filename}"
            case FileEvent.FILE_MODIFIED:
                old_size = data.get("old_size", "?")
                new_size = data.get("new_size", "?")
                message = f"[{timestamp} ✏️ File modified: {filename} ({old_size} -> {new_size})"
            case FileEvent.FILE_COPIED:
                destination = data.get("destination", "?")
                message = f"[{timestamp}] 📋 File copied: {filename} → {destination}"
            case FileEvent.FILE_MOVED:
                destination = data.get("destination", "?")
                message = f"[{timestamp}] 📋 File moved: {filename} → {destination}"
            case _:
                message = f"[{timestamp}] ❓ Unknown event '{event}' for {filename}"

        print(f"\n{message}")


# ============================================================================
# STRATEGY PATTERN - TODO: Implement sort and filter strategies!
# ============================================================================


class SortStrategy(ABC):
    """Base class for sorting strategies"""

    @abstractmethod
    def sort(self, files: List["File"], reverse=False) -> List["File"]:
        pass


class SortByNameStrategy(SortStrategy):
    """Sort files alphabetically by name"""

    def sort(self, files: List["File"], reverse=False) -> List["File"]:
        return sorted(files, key=lambda f: f.filename, reverse=reverse)


class SortBySizeStrategy(SortStrategy):
    """Sort files by size"""

    def sort(self, files: List["File"], reverse=False) -> List["File"]:
        return sorted(files, key=lambda f: f.size, reverse=reverse)


class SortByDateStrategy(SortStrategy):
    """Sort files by modification date"""

    def sort(self, files: List["File"], reverse=False) -> List["File"]:
        return sorted(files, key=lambda f: f.modified_date, reverse=reverse)


class FilterStrategy:
    """Base class for filtering strategies"""

    @abstractmethod
    def filter(self, files: List["File"]) -> List["File"]:
        pass


class ShowAllFilter(FilterStrategy):
    """Show all files (no filtering)"""

    def filter(self, files: List["File"]) -> List["File"]:
        return files


class FilterByExtensionStrategy(FilterStrategy):
    """Filter files by extension"""

    def __init__(self, extension: str):
        self.extension = extension

    def filter(self, files: List["File"]) -> List["File"]:
        return [f for f in files if f.filename.endswith(self.extension)]


class FilterBySizeStrategy(FilterStrategy):
    """Filter files by size range"""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def filter(self, files: List["File"]) -> List["File"]:
        return [f for f in files if self.min_size <= f.size <= self.max_size]


# ============================================================================
# FILE MODEL - Base file representation
# ============================================================================


class File:
    """Represents a file with metadata"""

    def __init__(self, filename: str, path: str, size: int, modified_date: datetime):
        self.filename = filename  # Changed!
        self.path = path
        self.size = size
        self.modified_date = modified_date

    def get_extension(self) -> str:
        """Get file extension"""
        return os.path.splitext(self.filename)[1]  # Changed!

    def __str__(self) -> str:
        return f"{self.filename} ({self.size} bytes)"  # Changed!


# ============================================================================
# FACTORY PATTERN - TODO: Implement file factory!
# ============================================================================


class FileTypeCat(Enum):
    TEXT = 1
    IMAGE = 2
    DOCUMENT = 3
    UNKNOWN = 999

    def display(self):
        """Return human-readable name"""
        return self.name.capitalize()


class FileType(ABC):
    """Base class for different file types"""

    def __init__(self, file: File):
        self.file = file

    @abstractmethod
    def get_type_name(self) -> FileTypeCat:
        pass

    @abstractmethod
    def get_icon(self) -> str:
        pass


class TextFile(FileType):
    """Text file (.txt, .md, .py, etc.)"""

    def get_type_name(self) -> FileTypeCat:
        return FileTypeCat.TEXT

    def get_icon(self) -> str:
        return "📄"


class ImageFile(FileType):
    """Image file (.jpg, .png, etc.)"""

    def get_type_name(self) -> FileTypeCat:
        return FileTypeCat.IMAGE

    def get_icon(self) -> str:
        return "🖼️"


class DocumentFile(FileType):
    """Document file (.pdf, .docx, etc.)"""

    def get_type_name(self) -> FileTypeCat:
        return FileTypeCat.DOCUMENT

    def get_icon(self) -> str:
        return "📃"


class UnknownFile(FileType):
    """Unknown file type"""

    def get_type_name(self) -> FileTypeCat:
        return FileTypeCat.UNKNOWN

    def get_icon(self) -> str:
        return "📦"


class FileFactory:
    """Factory to create appropriate file type based on extension"""

    TEXT_EXTENSIONS = [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"]
    DOCUMENT_EXTENSIONS = [".pdf", ".doc", ".docx", ".xls", ".xlsx"]

    @staticmethod
    def create_file_type(file: File) -> FileType:
        """
        Create appropriate file type based on extension
        """

        file_extension = file.get_extension()

        if file_extension in FileFactory.TEXT_EXTENSIONS:
            return TextFile(file)
        elif file_extension in FileFactory.IMAGE_EXTENSIONS:
            return ImageFile(file)
        elif file_extension in FileFactory.DOCUMENT_EXTENSIONS:
            return DocumentFile(file)
        else:
            return UnknownFile(file)


# ============================================================================
# DECORATOR PATTERN - TODO: Implement file decorators!
# ============================================================================


class FileDecorator(File):
    """Base decorator for files"""

    def __init__(self, file: File):
        self._file = file

    # Delegate all File properties to wrapped file
    @property
    def filename(self):
        return self._file.filename

    @property
    def path(self):
        return self._file.path

    @property
    def size(self):
        return self._file.size

    @property
    def modified_date(self):
        return self._file.modified_date

    def get_extension(self) -> str:
        return self._file.get_extension()


class TaggedFile(FileDecorator):
    """Decorator that adds tags to a file"""

    def __init__(self, file: File, tag: str):
        super().__init__(file)
        self.tag = tag

    def __str__(self) -> str:
        return f"{str(self._file)} [#{self.tag}]"


class FavoritedFile(FileDecorator):
    """Decorator that marks a file as favorite"""

    def __init__(self, file: File):
        super().__init__(file)
        self.is_favorite = True

    def __str__(self) -> str:
        return f"⭐ {str(self._file)}"


class DescribedFile(FileDecorator):
    """Decorator that adds a description to a file"""

    def __init__(self, file: File, description: str):
        super().__init__(file)
        self.description = description

    def __str__(self) -> str:
        return f"{str(self._file)} - {self.description}"


# ============================================================================
# COMMAND PATTERN - TODO: Implement file operation commands!
# ============================================================================


class Command(ABC):
    """Base command class"""

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass


class CopyFileCommand(Command):
    """Command to copy a file"""

    def __init__(self, source: str, destination: str):
        self.source = source
        self.destination = destination

    def execute(self):
        # For now, just print "Copied {source} to {destination}"
        # In real version, would use: shutil.copy(self.source, self.destination)
        print(f"Copied {self.source} to {self.destination}")

    def undo(self):
        # For now, just print "Removed {destination}"
        # In real version, would use: os.remove(self.destination)
        print(f"Removed {self.destination}")


class DeleteFileCommand(Command):
    """Command to delete a file"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_backup = None  # Store file content for undo

    def execute(self):
        # For now, just print "Deleted {filepath}"
        print(f"Deleted {self.filepath}")

    def undo(self):
        # For now, just print "Restored {filepath}"
        print(f"Restored {self.filepath}")


class RenameFileCommand(Command):
    """Command to rename a file"""

    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def execute(self):
        # For now, just print "Renamed {old_name} to {new_name}"
        print(f"Renamed {self.old_name} to {self.new_name}")

    def undo(self):
        # For now, just print "Renamed {new_name} back to {old_name}"
        print(f"Renamed {self.new_name} back to {self.old_name}")


class CommandHistory:
    """Manages command history for undo/redo"""

    def __init__(self):
        self.history: List[Command] = []  # Commands that have been executed
        self.redo_stack: List[Command] = []  # Commands that have been undone

    def execute_command(self, command: Command):
        """Execute a command and add it to history"""
        command.execute()
        self.history.append(command)

        # Clear redo stack when a new command is executed
        # (You can't redo after doing something new!)
        self.redo_stack.clear()

    def undo(self) -> bool:
        """Undo the last command"""
        if not self.history:
            return False

        # Pop from history
        command = self.history.pop()

        # Undo it
        command.undo()

        # Save to redo stack
        self.redo_stack.append(command)

        return True

    def redo(self) -> bool:
        """Redo the last undone command"""
        if not self.redo_stack:
            return False

        # Pop from redo stack
        command = self.redo_stack.pop()

        # Re-execute it
        command.execute()

        # Add back to history
        self.history.append(command)

        return True

    def can_undo(self) -> bool:
        """Check if there are commands to undo"""
        return len(self.history) > 0

    def can_redo(self) -> bool:
        """Check if there are commands to redo"""
        return len(self.redo_stack) > 0

    def get_history_summary(self) -> str:
        """Get a summary of command history"""
        return f"History: {len(self.history)} commands, Redo: {len(self.redo_stack)}"


# ============================================================================
# FILE MANAGER - Main class (uses Subject for Observer pattern)
# ============================================================================


class FileManager(Subject):
    """
    Main file manager class
    Uses all 5 patterns!
    """
    
    def __init__(self):
        super().__init__()
        self.files: List[File] = []
        self.command_history: CommandHistory = CommandHistory()
        self.sort_strategy: SortStrategy = SortByNameStrategy()
        self.filter_strategy: FilterStrategy = ShowAllFilter()
    
    def set_sort_strategy(self, strategy: SortStrategy):
        self.sort_strategy = strategy
    
    def set_filter_strategy(self, strategy: FilterStrategy):
        self.filter_strategy = strategy
    
    def add_file(self, file: File):
        """Add a file to the manager"""
        self.files.append(file)
        self.notify(FileEvent.FILE_ADDED, {
            "filename": file.filename,
            "size": file.size,
        })
    
    def remove_file(self, filename: str):
        """Remove a file from the manager"""
        file = self.get_file(filename)
        if file:
            self.files.remove(file)
            self.notify(FileEvent.FILE_DELETED, {
                "filename": file.filename,
            })
    
    def get_file(self, filename: str) -> Optional[File]:
        """Get a file by name"""
        for file in self.files:
            if file.filename == filename:
                return file
        return None
    
    def list_files(self):
        """List files with current sort and filter strategies applied"""
        filtered_files = self.filter_strategy.filter(self.files)
        sorted_files = self.sort_strategy.sort(filtered_files)
        
        if not sorted_files:
            print("No files to display")
        else:
            for file in sorted_files:
                print(file)
    
    def copy_file(self, source: str, destination: str):
        """Copy a file with undo support"""
        command = CopyFileCommand(source, destination)
        self.command_history.execute_command(command)
    
    def delete_file(self, filepath: str):
        """Delete a file with undo support"""
        command = DeleteFileCommand(filepath)
        self.command_history.execute_command(command)
    
    def rename_file(self, old_name: str, new_name: str):
        """Rename a file with undo support"""
        command = RenameFileCommand(old_name, new_name)
        self.command_history.execute_command(command)
    
    def undo(self):
        """Undo last operation"""
        if self.command_history.can_undo():
            self.command_history.undo()
    
    def redo(self):
        """Redo last undone operation"""
        if self.command_history.can_redo():
            self.command_history.redo()


# ============================================================================
# DEMO / TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎯 FILE MANAGER - All 5 Design Patterns Demo")
    print("=" * 70)
    
    # ========================================================================
    # 1. OBSERVER PATTERN - Attach observer to get notifications
    # ========================================================================
    print("\n📋 1. OBSERVER PATTERN - Setting up notifications")
    print("-" * 70)
    
    manager = FileManager()
    observer = FileChangeObserver()
    manager.attach(observer)
    print("✅ Observer attached to FileManager")
    
    # ========================================================================
    # 2. FACTORY PATTERN - Create different file types
    # ========================================================================
    print("\n\n🏭 2. FACTORY PATTERN - Creating files with factory")
    print("-" * 70)
    
    # Create base files
    file1 = File("report.txt", "/home/docs", 1024, datetime.now())
    file2 = File("photo.jpg", "/home/pics", 2048, datetime.now())
    file3 = File("presentation.pdf", "/home/work", 4096, datetime.now())
    file4 = File("script.py", "/home/code", 512, datetime.now())
    
    # Use factory to get file types
    type1 = FileFactory.create_file_type(file1)
    type2 = FileFactory.create_file_type(file2)
    type3 = FileFactory.create_file_type(file3)
    type4 = FileFactory.create_file_type(file4)
    
    print(f"{type1.get_icon()} {file1.filename} - Type: {type1.get_type_name().name}")
    print(f"{type2.get_icon()} {file2.filename} - Type: {type2.get_type_name().name}")
    print(f"{type3.get_icon()} {file3.filename} - Type: {type3.get_type_name().name}")
    print(f"{type4.get_icon()} {file4.filename} - Type: {type4.get_type_name().name}")
    
    # ========================================================================
    # 3. Add files to manager (triggers OBSERVER notifications!)
    # ========================================================================
    print("\n\n📁 3. Adding files to manager (Observer will notify)")
    print("-" * 70)
    
    manager.add_file(file1)
    manager.add_file(file2)
    manager.add_file(file3)
    manager.add_file(file4)
    
    # ========================================================================
    # 4. DECORATOR PATTERN - Add metadata to files
    # ========================================================================
    print("\n\n🎨 4. DECORATOR PATTERN - Adding metadata to files")
    print("-" * 70)
    
    # Decorate file1 with tag
    tagged_file = TaggedFile(file1, "urgent")
    print(f"Tagged: {tagged_file}")
    
    # Stack multiple decorators!
    stacked_file = TaggedFile(file2, "important")
    stacked_file = FavoritedFile(stacked_file)
    stacked_file = DescribedFile(stacked_file, "Vacation photo")
    print(f"Stacked decorators: {stacked_file}")
    
    # ========================================================================
    # 5. STRATEGY PATTERN - Sort and filter files
    # ========================================================================
    print("\n\n🔀 5. STRATEGY PATTERN - Sorting and filtering")
    print("-" * 70)
    
    print("\n📋 Default view (sorted by name):")
    manager.list_files()
    
    # Change sort strategy
    print("\n📋 Sorted by size (largest first):")
    manager.set_sort_strategy(SortBySizeStrategy())
    manager.list_files()
    
    # Change filter strategy
    print("\n📋 Filter by extension (.txt files only):")
    manager.set_filter_strategy(FilterByExtensionStrategy(".txt"))
    manager.list_files()
    
    # Reset to show all
    print("\n📋 Back to showing all files, sorted by date:")
    manager.set_filter_strategy(ShowAllFilter())
    manager.set_sort_strategy(SortByDateStrategy())
    manager.list_files()
    
    # ========================================================================
    # 6. COMMAND PATTERN - File operations with undo/redo
    # ========================================================================
    print("\n\n⚡ 6. COMMAND PATTERN - File operations with undo/redo")
    print("-" * 70)
    
    print("\n📋 Executing commands:")
    manager.copy_file("report.txt", "report_backup.txt")
    manager.rename_file("photo.jpg", "vacation.jpg")
    manager.delete_file("old_file.txt")
    
    print("\n↩️  Undoing last command:")
    manager.undo()
    
    print("\n↩️  Undoing another command:")
    manager.undo()
    
    print("\n↪️  Redoing last undone command:")
    manager.redo()
    
    print("\n📋 Executing new command (this clears redo stack):")
    manager.copy_file("script.py", "script_backup.py")
    
    print("\n❌ Trying to redo (should fail - redo stack was cleared):")
    if manager.command_history.can_redo():
        manager.redo()
    else:
        print("⚠️  Cannot redo - redo stack is empty")
    
    # ========================================================================
    # 7. ALL PATTERNS TOGETHER - Final demo
    # ========================================================================
    print("\n\n🌟 7. ALL PATTERNS WORKING TOGETHER")
    print("-" * 70)
    
    # Create a new file
    new_file = File("data.json", "/home/data", 256, datetime.now())
    
    # Use Factory to identify type
    file_type = FileFactory.create_file_type(new_file)
    print(f"\nFactory created: {file_type.get_icon()} {new_file.filename} ({file_type.get_type_name().name})")
    
    # Decorate it
    decorated = TaggedFile(new_file, "config")
    decorated = FavoritedFile(decorated)
    print(f"After decoration: {decorated}")
    
    # Add to manager (Observer will notify)
    print("\nAdding decorated file to manager:")
    manager.add_file(decorated)
    
    # Filter and sort (Strategy)
    print("\nFiltering for favorited files and sorting by name:")
    # Note: Our current filter strategies don't detect decorators
    # But we can still list all files
    manager.set_filter_strategy(ShowAllFilter())
    manager.set_sort_strategy(SortByNameStrategy())
    manager.list_files()
    
    # Use command pattern to manipulate
    print("\nUsing commands to copy file:")
    manager.copy_file("data.json", "data_backup.json")
    
    print("\nUndoing copy:")
    manager.undo()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("✨ SUMMARY - Patterns Demonstrated:")
    print("=" * 70)
    print("✅ Observer Pattern:  File change notifications")
    print("✅ Strategy Pattern:  Flexible sorting and filtering")
    print("✅ Factory Pattern:   Create file types by extension")
    print("✅ Decorator Pattern: Add metadata dynamically (tags, favorites, etc.)")
    print("✅ Command Pattern:   File operations with undo/redo support")
    print("\n🎉 All 5 patterns working together in harmony!")
    print("=" * 70)