"""
Task Manager - Phase 7: Singleton Pattern
Learn how to implement the Singleton pattern for single-instance classes

Patterns Implemented:
1. Observer Pattern - Event notifications
2. Strategy Pattern - Sorting/filtering/display
3. Command Pattern - Undo/redo
4. Factory Pattern - Task creation
5. Decorator Pattern - Dynamic feature addition
6. Singleton Pattern - Single instance classes (NEW!)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Set, Dict, Any
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import threading


# ============================================================================
# SINGLETON PATTERN - BASE IMPLEMENTATION
# ============================================================================


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.

    This is the SINGLETON PATTERN in action!
    - Controls class instantiation
    - Ensures only one instance exists
    - Thread-safe with locks
    - Returns same instance on repeated calls
    """

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Called when someone tries to create an instance.
        Returns existing instance or creates new one.
        """
        # Double-checked locking for thread safety
        if cls not in cls._instances:
            with cls._lock:
                # Check again inside lock
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance

        return cls._instances[cls]


# ============================================================================
# SINGLETON PATTERN - CONFIG MANAGER
# ============================================================================


class ConfigManager(metaclass=SingletonMeta):
    """
    Singleton configuration manager.
    Stores application-wide settings.

    Only ONE instance can exist!
    """

    def __init__(self):
        # Only runs once (first instantiation)
        if not hasattr(self, "_initialized"):
            self._config: Dict[str, Any] = {
                "urgent_deadline_hours": 24,
                "max_undo_history": 50,
                "enable_notifications": True,
                "date_format": "%Y-%m-%d %H:%M",
                "auto_save": False,
                "theme": "default",
            }
            self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()

    def reset(self):
        """Reset to default configuration"""
        # Recreate the config dict so reset actually works for the Singleton.
        self._config = {
            "urgent_deadline_hours": 24,
            "max_undo_history": 50,
            "enable_notifications": True,
            "date_format": "%Y-%m-%d %H:%M",
            "auto_save": False,
            "theme": "default",
        }

    def __repr__(self):
        return f"ConfigManager(id={id(self)}, configs={len(self._config)})"


# ============================================================================
# SINGLETON PATTERN - TASK LOGGER
# ============================================================================


class TaskLogger(metaclass=SingletonMeta):
    """
    Singleton logger for task operations.
    Centralized logging system.

    Only ONE logger instance exists!
    """

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._logs: List[Dict[str, Any]] = []
            self._max_logs = 1000
            self._enabled = True
            self._initialized = True

    def log(self, level: str, message: str, **kwargs):
        """Log a message"""
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now(),
            "level": level.upper(),
            "message": message,
            "data": kwargs,
        }

        self._logs.append(entry)

        # Keep only recent logs
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log("DEBUG", message, **kwargs)

    def get_logs(
        self, level: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent logs"""
        logs = self._logs

        if level:
            logs = [log for log in logs if log["level"] == level.upper()]

        return logs[-limit:]

    def clear(self):
        """Clear all logs"""
        self._logs.clear()

    def enable(self):
        """Enable logging"""
        self._enabled = True

    def disable(self):
        """Disable logging"""
        self._enabled = False

    def __repr__(self):
        return f"TaskLogger(id={id(self)}, logs={len(self._logs)})"


# ============================================================================
# SINGLETON PATTERN - GLOBAL STATISTICS
# ============================================================================


class GlobalStats(metaclass=SingletonMeta):
    """
    Singleton for tracking global application statistics.

    Only ONE stats tracker exists!
    """

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._stats = {
                "tasks_created": 0,
                "tasks_completed": 0,
                "tasks_deleted": 0,
                "commands_executed": 0,
                "undos_performed": 0,
                "redos_performed": 0,
                "decorations_applied": 0,
                "app_start_time": datetime.now(),
            }
            self._initialized = True

    def increment(self, stat_name: str, amount: int = 1):
        """Increment a statistic"""
        if stat_name in self._stats:
            self._stats[stat_name] += amount

    def get(self, stat_name: str) -> Any:
        """Get a statistic"""
        return self._stats.get(stat_name, 0)

    def get_all(self) -> Dict[str, Any]:
        """Get all statistics"""
        stats = self._stats.copy()
        # Add runtime
        runtime = datetime.now() - stats["app_start_time"]
        stats["runtime_seconds"] = int(runtime.total_seconds())
        return stats

    def reset(self):
        """Reset all statistics"""
        for key in self._stats:
            if key != "app_start_time":
                self._stats[key] = 0

    def __repr__(self):
        return f"GlobalStats(id={id(self)}, stats={len(self._stats)})"


# ============================================================================
# TASK TYPES (From Previous Phases)
# ============================================================================


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class Task(ABC):
    def __init__(self, title: str, description: str = ""):
        if not title.strip():
            raise ValueError("Task title cannot be empty")

        self.id = self._generate_id()
        self.title = title.strip()
        self.description = description.strip()
        self.completed = False
        self.created_at = datetime.now()

        # Log task creation using Singleton logger
        TaskLogger().info(f"Task created: {self.title}", task_id=self.id)
        GlobalStats().increment("tasks_created")

    @staticmethod
    def _generate_id() -> str:
        return str(int(datetime.now().timestamp() * 1000000))

    @abstractmethod
    def get_priority(self) -> TaskPriority:
        pass

    @abstractmethod
    def get_type_name(self) -> str:
        pass

    @abstractmethod
    def get_display_icon(self) -> str:
        pass

    def toggle_completion(self):
        self.completed = not self.completed
        if self.completed:
            TaskLogger().info(f"Task completed: {self.title}", task_id=self.id)
            GlobalStats().increment("tasks_completed")

    def get_full_description(self) -> str:
        return self.description

    def get_tags(self) -> Set[str]:
        return set()

    def is_type(self, task_class) -> bool:
        return isinstance(self, task_class)

    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        icon = self.get_display_icon()
        return f"{status} {icon} {self.title}"


class NormalTask(Task):
    def get_priority(self) -> TaskPriority:
        return TaskPriority.NORMAL

    def get_type_name(self) -> str:
        return "Normal"

    def get_display_icon(self) -> str:
        return "📝"


class UrgentTask(Task):
    def __init__(
        self, title: str, description: str = "", deadline: Optional[datetime] = None
    ):
        # Get default from Singleton config!
        config = ConfigManager()
        default_hours = config.get("urgent_deadline_hours", 24)

        super().__init__(title, description)

        if deadline is None:
            self.deadline = datetime.now() + timedelta(hours=default_hours)
        else:
            self.deadline = deadline

    def get_priority(self) -> TaskPriority:
        return TaskPriority.URGENT

    def get_type_name(self) -> str:
        return "Urgent"

    def get_display_icon(self) -> str:
        return "🔥"

    def is_overdue(self) -> bool:
        return datetime.now() > self.deadline and not self.completed

    def __str__(self) -> str:
        base = super().__str__()
        config = ConfigManager()
        date_format = config.get("date_format", "%Y-%m-%d %H:%M")

        if self.is_overdue():
            return f"{base} ⚠️ OVERDUE"
        return f"{base} (Due: {self.deadline.strftime(date_format)})"


class RecurringTask(Task):
    def __init__(self, title: str, description: str = "", frequency_days: int = 7):
        super().__init__(title, description)
        self.frequency_days = frequency_days
        self.next_occurrence = datetime.now() + timedelta(days=frequency_days)

    def get_priority(self) -> TaskPriority:
        return TaskPriority.NORMAL

    def get_type_name(self) -> str:
        return "Recurring"

    def get_display_icon(self) -> str:
        return "🔄"

    def toggle_completion(self):
        if not self.completed:
            self.completed = True
            self.next_occurrence = datetime.now() + timedelta(days=self.frequency_days)
            GlobalStats().increment("tasks_completed")
        else:
            self.completed = False

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (Every {self.frequency_days} days)"


class ImportantTask(Task):
    def get_priority(self) -> TaskPriority:
        return TaskPriority.HIGH

    def get_type_name(self) -> str:
        return "Important"

    def get_display_icon(self) -> str:
        return "⭐"


# ============================================================================
# DECORATOR PATTERN (From Phase 6)
# ============================================================================


class TaskDecorator(Task):
    def __init__(self, task: Task):
        self._wrapped_task = task
        GlobalStats().increment("decorations_applied")

    @property
    def id(self):
        return self._wrapped_task.id

    @property
    def title(self):
        return self._wrapped_task.title

    @property
    def description(self):
        return self._wrapped_task.description

    @property
    def completed(self):
        return self._wrapped_task.completed

    @completed.setter
    def completed(self, value):
        self._wrapped_task.completed = value

    @property
    def created_at(self):
        return self._wrapped_task.created_at

    def get_priority(self) -> TaskPriority:
        return self._wrapped_task.get_priority()

    def get_type_name(self) -> str:
        return self._wrapped_task.get_type_name()

    def get_display_icon(self) -> str:
        return self._wrapped_task.get_display_icon()

    def toggle_completion(self):
        self._wrapped_task.toggle_completion()

    def get_full_description(self) -> str:
        return self._wrapped_task.get_full_description()

    def get_tags(self) -> Set[str]:
        return self._wrapped_task.get_tags()

    def is_type(self, task_class) -> bool:
        if isinstance(self, task_class):
            return True
        return self._wrapped_task.is_type(task_class)

    def __str__(self) -> str:
        return str(self._wrapped_task)


class PriorityBoostDecorator(TaskDecorator):
    def __init__(self, task: Task, boost_levels: int = 1):
        super().__init__(task)
        self.boost_levels = boost_levels

    def get_priority(self) -> TaskPriority:
        base_priority = self._wrapped_task.get_priority()
        boosted_value = min(
            base_priority.value + self.boost_levels, TaskPriority.CRITICAL.value
        )
        return TaskPriority(boosted_value)

    def get_display_icon(self) -> str:
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}🔺"

    def __str__(self) -> str:
        base = str(self._wrapped_task)
        return f"{base} [PRIORITY BOOSTED]"


class ReminderDecorator(TaskDecorator):
    def __init__(self, task: Task, reminder_time: datetime):
        super().__init__(task)
        self.reminder_time = reminder_time

    def is_reminder_due(self) -> bool:
        return datetime.now() >= self.reminder_time and not self.completed

    def get_display_icon(self) -> str:
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}🔔"

    def __str__(self) -> str:
        base = str(self._wrapped_task)
        config = ConfigManager()
        date_format = config.get("date_format", "%Y-%m-%d %H:%M")

        if self.is_reminder_due():
            return f"{base} 🔔 REMINDER!"
        return f"{base} (Reminder: {self.reminder_time.strftime(date_format)})"


class TagDecorator(TaskDecorator):
    def __init__(self, task: Task, tag: str):
        super().__init__(task)
        self.tag = tag

    def get_tags(self) -> Set[str]:
        tags = self._wrapped_task.get_tags()
        tags.add(self.tag)
        return tags

    def get_display_icon(self) -> str:
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}🏷️"

    def __str__(self) -> str:
        base = str(self._wrapped_task)
        all_tags = self.get_tags()
        if all_tags:
            tags_str = " ".join(f"#{tag}" for tag in sorted(all_tags))
            return f"{base} [{tags_str}]"
        return base


class NotesDecorator(TaskDecorator):
    def __init__(self, task: Task, note: str):
        super().__init__(task)
        self.note = note

    def get_full_description(self) -> str:
        base_desc = self._wrapped_task.get_full_description()
        if base_desc:
            return f"{base_desc}\n\nNote: {self.note}"
        return f"Note: {self.note}"

    def get_display_icon(self) -> str:
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}📎"


class DelegatedDecorator(TaskDecorator):
    def __init__(self, task: Task, delegated_to: str):
        super().__init__(task)
        self.delegated_to = delegated_to

    def get_display_icon(self) -> str:
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}👥"

    def __str__(self) -> str:
        base = str(self._wrapped_task)
        return f"{base} (Delegated to: {self.delegated_to})"


# ============================================================================
# FACTORY PATTERN (From Phase 5)
# ============================================================================


class TaskFactory:
    URGENT_KEYWORDS = ["urgent", "asap", "critical", "emergency", "immediately"]
    RECURRING_KEYWORDS = ["daily", "weekly", "monthly", "every", "recurring"]
    IMPORTANT_KEYWORDS = ["important", "priority", "key", "crucial", "vital"]

    @staticmethod
    def create_task(
        title: str,
        description: str = "",
        task_type: str = "auto",
        deadline: Optional[datetime] = None,
        frequency_days: int = 7,
    ) -> Task:

        if task_type == "auto":
            task_type = TaskFactory._detect_task_type(title, description)

        if task_type == "urgent":
            return UrgentTask(title, description, deadline)
        elif task_type == "recurring":
            return RecurringTask(title, description, frequency_days)
        elif task_type == "important":
            return ImportantTask(title, description)
        else:
            return NormalTask(title, description)

    @staticmethod
    def _detect_task_type(title: str, description: str) -> str:
        text = (title + " " + description).lower()

        if any(keyword in text for keyword in TaskFactory.URGENT_KEYWORDS):
            return "urgent"
        if any(keyword in text for keyword in TaskFactory.RECURRING_KEYWORDS):
            return "recurring"
        if any(keyword in text for keyword in TaskFactory.IMPORTANT_KEYWORDS):
            return "important"

        return "normal"


# ============================================================================
# OBSERVER PATTERN (From Phase 2)
# ============================================================================


class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: dict):
        pass


class NotificationObserver(Observer):
    def update(self, event: str, data: dict):
        config = ConfigManager()

        if not config.get("enable_notifications", True):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        if event == "task_added":
            task_type = data.get("type", "task")
            message = f"🔔 [{timestamp}] New {task_type} added: '{data['title']}'"
        elif event == "task_decorated":
            decorator_type = data.get("decorator", "decoration")
            message = f"🔔 [{timestamp}] Task decorated with {decorator_type}: '{data['title']}'"
        elif event == "task_completed":
            message = f"🔔 [{timestamp}] Task completed: '{data['title']}'"
        elif event == "task_uncompleted":
            message = f"🔔 [{timestamp}] Task reopened: '{data['title']}'"
        elif event == "task_deleted":
            message = f"🔔 [{timestamp}] Task deleted: '{data['title']}'"
        elif event == "command_executed":
            message = f"🔔 [{timestamp}] {data['command']}"
        elif event == "command_undone":
            message = f"🔔 [{timestamp}] Undone: {data['command']}"
        else:
            return

        print(f"\n{message}")


class Subject:
    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify(self, event: str, data: dict):
        for observer in self._observers:
            observer.update(event, data)


# ============================================================================
# STRATEGY PATTERN (From Phase 3)
# ============================================================================


class SortStrategy(ABC):
    @abstractmethod
    def sort(self, tasks: List[Task]) -> List[Task]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class SortByPriorityStrategy(SortStrategy):
    def sort(self, tasks: List[Task]) -> List[Task]:
        return sorted(tasks, key=lambda t: t.get_priority().value, reverse=True)

    def name(self) -> str:
        return "Priority (Highest First)"


class SortByDateStrategy(SortStrategy):
    def sort(self, tasks: List[Task]) -> List[Task]:
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def name(self) -> str:
        return "Date (Newest First)"


class FilterStrategy(ABC):
    @abstractmethod
    def filter(self, tasks: List[Task]) -> List[Task]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class ShowAllFilter(FilterStrategy):
    def filter(self, tasks: List[Task]) -> List[Task]:
        return tasks

    def name(self) -> str:
        return "All Tasks"


class ShowUrgentFilter(FilterStrategy):
    def filter(self, tasks: List[Task]) -> List[Task]:
        return [t for t in tasks if t.is_type(UrgentTask)]

    def name(self) -> str:
        return "Urgent Only"


class ShowPendingFilter(FilterStrategy):
    def filter(self, tasks: List[Task]) -> List[Task]:
        return [t for t in tasks if not t.completed]

    def name(self) -> str:
        return "Pending Only"


class DisplayStrategy(ABC):
    @abstractmethod
    def display(self, tasks: List[Task]):
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class SimpleDisplayStrategy(DisplayStrategy):
    def display(self, tasks: List[Task]):
        if not tasks:
            print("No tasks to display.")
            return

        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")

    def name(self) -> str:
        return "Simple"


class DetailedDisplayStrategy(DisplayStrategy):
    def display(self, tasks: List[Task]):
        if not tasks:
            print("No tasks to display.")
            return

        config = ConfigManager()
        date_format = config.get("date_format", "%Y-%m-%d %H:%M")

        for i, task in enumerate(tasks, 1):
            status = "✓ DONE" if task.completed else "○ PENDING"
            print(f"\n{i}. [{task.id}] {status}")
            print(f"   Type: {task.get_type_name()} {task.get_display_icon()}")
            print(f"   Priority: {task.get_priority().name}")
            print(f"   Title: {task.title}")

            full_desc = task.get_full_description()
            if full_desc:
                print(f"   Description: {full_desc}")

            tags = task.get_tags()
            if tags:
                print(f"   Tags: {', '.join(f'#{tag}' for tag in sorted(tags))}")

            print(f"   Created: {task.created_at.strftime(date_format)}")

            if isinstance(task, TaskDecorator):
                print(f"   📦 Decorated Task")

    def name(self) -> str:
        return "Detailed"


# ============================================================================
# COMMAND PATTERN (From Phase 4)
# ============================================================================


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass


class AddTaskCommand(Command):
    def __init__(
        self,
        task_manager: "TaskManager",
        title: str,
        description: str = "",
        task_type: str = "auto",
        deadline: Optional[datetime] = None,
        frequency_days: int = 7,
    ):
        self.task_manager = task_manager
        self.title = title
        self.description = description
        self.task_type = task_type
        self.deadline = deadline
        self.frequency_days = frequency_days
        self.task = None

    def execute(self):
        self.task = self.task_manager._add_task_internal(
            self.title,
            self.description,
            self.task_type,
            self.deadline,
            self.frequency_days,
        )
        self.task_manager.notify(
            "command_executed", {"command": self.get_description()}
        )
        GlobalStats().increment("commands_executed")

    def undo(self):
        if self.task:
            self.task_manager._delete_task_internal(self.task.id)
            self.task_manager.notify(
                "command_undone", {"command": self.get_description()}
            )
            GlobalStats().increment("undos_performed")

    def get_description(self) -> str:
        type_name = self.task.get_type_name() if self.task else "Task"
        return f"Add {type_name}: '{self.title}'"


class DeleteTaskCommand(Command):
    def __init__(self, task_manager: "TaskManager", task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id
        self.deleted_task = None
        self.deleted_index = None

    def execute(self):
        task = self.task_manager.get_task_by_id(self.task_id)
        if task:
            self.deleted_index = self.task_manager.tasks.index(task)
            self.deleted_task = deepcopy(task)
            self.task_manager._delete_task_internal(self.task_id)
            self.task_manager.notify(
                "command_executed", {"command": self.get_description()}
            )
            GlobalStats().increment("commands_executed")

    def undo(self):
        if self.deleted_task is not None:
            if self.deleted_index is not None and 0 <= self.deleted_index <= len(
                self.task_manager.tasks
            ):
                self.task_manager.tasks.insert(self.deleted_index, self.deleted_task)
            else:
                self.task_manager.tasks.append(self.deleted_task)
            self.task_manager.notify(
                "command_undone", {"command": self.get_description()}
            )
            GlobalStats().increment("undos_performed")


class ToggleTaskCommand(Command):
    def __init__(self, task_manager: "TaskManager", task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id
        self.task = None

    def execute(self):
        self.task = self.task_manager.get_task_by_id(self.task_id)
        if self.task:
            self.task.toggle_completion()

            event = "task_completed" if self.task.completed else "task_uncompleted"
            self.task_manager.notify(
                event, {"id": self.task.id, "title": self.task.title}
            )
            self.task_manager.notify(
                "command_executed", {"command": self.get_description()}
            )
            GlobalStats().increment("commands_executed")

    def undo(self):
        if self.task:
            self.task.toggle_completion()

            event = "task_completed" if self.task.completed else "task_uncompleted"
            self.task_manager.notify(
                event, {"id": self.task.id, "title": self.task.title}
            )
            self.task_manager.notify(
                "command_undone", {"command": self.get_description()}
            )
            GlobalStats().increment("undos_performed")

    def get_description(self) -> str:
        if self.task:
            status = "Complete" if self.task.completed else "Uncomplete"
            return f"{status} Task: '{self.task.title}'"
        return "Toggle Task"


class DecorateTaskCommand(Command):
    def __init__(
        self, task_manager: "TaskManager", task_id: str, decorator_type: str, **kwargs
    ):
        self.task_manager = task_manager
        self.task_id = task_id
        self.decorator_type = decorator_type
        self.kwargs = kwargs
        self.original_task_index = None

    def execute(self):
        task = self.task_manager.get_task_by_id(self.task_id)
        if not task:
            return

        self.original_task_index = self.task_manager.tasks.index(task)

        # Build the correct decorator instance based on decorator_type
        if self.decorator_type == "priority_boost":
            decorated = PriorityBoostDecorator(task, self.kwargs.get("boost_levels", 1))
        elif self.decorator_type == "reminder":
            reminder_time = self.kwargs.get("reminder_time")
            if reminder_time is None:
                return  # invalid usage; don't modify tasks
            decorated = ReminderDecorator(task, reminder_time)
        elif self.decorator_type == "tag":
            tag = self.kwargs.get("tag")
            if not tag:
                return
            decorated = TagDecorator(task, tag)
        elif self.decorator_type == "notes":
            note = self.kwargs.get("note")
            if not note:
                return
            decorated = NotesDecorator(task, note)
        elif self.decorator_type == "delegate":
            delegated_to = self.kwargs.get("delegated_to")
            if not delegated_to:
                return
            decorated = DelegatedDecorator(task, delegated_to)
        else:
            # Unknown decorator type — do nothing
            return

        self.task_manager.tasks[self.original_task_index] = decorated
        self.task_manager.notify(
            "task_decorated", {"title": task.title, "decorator": self.decorator_type}
        )
        GlobalStats().increment("commands_executed")

    def undo(self):
        if self.original_task_index is not None:
            decorated_task = self.task_manager.tasks[self.original_task_index]
            if isinstance(decorated_task, TaskDecorator):
                self.task_manager.tasks[self.original_task_index] = (
                    decorated_task._wrapped_task
                )
            GlobalStats().increment("undos_performed")

    def get_description(self) -> str:
        return f"Decorate task with {self.decorator_type}"


class CommandHistory:
    def __init__(self):
        self.history: List[Command] = []
        self.undo_stack: List[Command] = []
        # Get max history from Singleton config
        config = ConfigManager()
        self.max_history = config.get("max_undo_history", 50)

    def execute_command(self, command: Command):
        command.execute()
        self.history.append(command)
        self.undo_stack.clear()

        # Keep history limited
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def undo(self) -> bool:
        if not self.history:
            return False
        command = self.history.pop()
        command.undo()
        self.undo_stack.append(command)
        return True

    def redo(self) -> bool:
        if not self.undo_stack:
            return False
        command = self.undo_stack.pop()
        command.execute()
        self.history.append(command)
        GlobalStats().increment("redos_performed")
        return True

    def can_undo(self) -> bool:
        return len(self.history) > 0

    def can_redo(self) -> bool:
        return len(self.undo_stack) > 0


# TASK MANAGER
class TaskManager(Subject):
    def __init__(self):
        super().__init__()
        self.tasks: List[Task] = []
        self.command_history = CommandHistory()
        self.factory = TaskFactory()
        self.sort_strategy: SortStrategy = SortByPriorityStrategy()
        self.filter_strategy: FilterStrategy = ShowAllFilter()
        self.display_strategy: DisplayStrategy = SimpleDisplayStrategy()

        TaskLogger().info("TaskManager initialized")

    def set_sort_strategy(self, strategy: SortStrategy):
        self.sort_strategy = strategy

    def set_filter_strategy(self, strategy: FilterStrategy):
        self.filter_strategy = strategy

    def set_display_strategy(self, strategy: DisplayStrategy):
        self.display_strategy = strategy

    def add_task(
        self,
        title: str,
        description: str = "",
        task_type: str = "auto",
        deadline: Optional[datetime] = None,
        frequency_days: int = 7,
    ) -> bool:
        if not title.strip():
            return False

        command = AddTaskCommand(
            self, title, description, task_type, deadline, frequency_days
        )
        self.command_history.execute_command(command)
        return True

    def delete_task(self, task_id: str) -> bool:
        if not self.get_task_by_id(task_id):
            return False
        command = DeleteTaskCommand(self, task_id)
        self.command_history.execute_command(command)
        return True

    def toggle_task(self, task_id: str) -> bool:
        if not self.get_task_by_id(task_id):
            return False
        command = ToggleTaskCommand(self, task_id)
        self.command_history.execute_command(command)
        return True

    def decorate_task(self, task_id: str, decorator_type: str, **kwargs) -> bool:
        if not self.get_task_by_id(task_id):
            return False
        command = DecorateTaskCommand(self, task_id, decorator_type, **kwargs)
        self.command_history.execute_command(command)
        return True

    def undo(self) -> bool:
        return self.command_history.undo()

    def redo(self) -> bool:
        return self.command_history.redo()

    def _add_task_internal(
        self,
        title: str,
        description: str,
        task_type: str,
        deadline: Optional[datetime],
        frequency_days: int,
    ) -> Task:
        task = self.factory.create_task(
            title, description, task_type, deadline, frequency_days
        )
        self.tasks.append(task)
        self.notify(
            "task_added",
            {
                "id": task.id,
                "title": task.title,
                "type": task.get_type_name(),
                "description": task.description,
            },
        )
        return task

    def _delete_task_internal(self, task_id: str) -> bool:
        task = self.get_task_by_id(task_id)
        if task:
            self.notify("task_deleted", {"id": task.id, "title": task.title})
            self.tasks.remove(task)
            GlobalStats().increment("tasks_deleted")
            return True
        return False

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_tasks_with_strategies(self) -> List[Task]:
        filtered = self.filter_strategy.filter(self.tasks)
        sorted_tasks = self.sort_strategy.sort(filtered)
        return sorted_tasks

    def display_tasks(self):
        tasks = self.get_tasks_with_strategies()
        print(
            f"\n--- Tasks ({self.filter_strategy.name()}, {self.sort_strategy.name()}) ---"
        )
        self.display_strategy.display(tasks)

    def get_stats(self) -> dict:
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.completed)
        pending = total - completed

        urgent_count = sum(1 for t in self.tasks if t.is_type(UrgentTask))
        recurring_count = sum(1 for t in self.tasks if t.is_type(RecurringTask))
        important_count = sum(1 for t in self.tasks if t.is_type(ImportantTask))
        normal_count = sum(1 for t in self.tasks if t.is_type(NormalTask))
        decorated_count = sum(1 for t in self.tasks if isinstance(t, TaskDecorator))

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "completion_rate": (completed / total * 100) if total > 0 else 0,
            "urgent": urgent_count,
            "recurring": recurring_count,
            "important": important_count,
            "normal": normal_count,
            "decorated": decorated_count,
        }


# CLI INTERFACE
class TaskManagerCLI:
    def __init__(self):
        self.manager = TaskManager()
        notification_observer = NotificationObserver()
        self.manager.attach(notification_observer)

        self.sort_strategies = {
            "1": SortByPriorityStrategy(),
            "2": SortByDateStrategy(),
        }

        self.filter_strategies = {
            "1": ShowAllFilter(),
            "2": ShowUrgentFilter(),
            "3": ShowPendingFilter(),
        }

        self.display_strategies = {
            "1": SimpleDisplayStrategy(),
            "2": DetailedDisplayStrategy(),
        }

        self.running = True

        TaskLogger().info("TaskManagerCLI started")

    def display_menu(self):
        print("\n" + "=" * 75)
        print("📋 TASK MANAGER - Phase 7: Singleton Pattern")
        print("=" * 75)
        print("Tasks:")
        print("  1. Add Task")
        print("  2. View Tasks")
        print("  3. Complete/Uncomplete Task")
        print("  4. Delete Task")
        print("\nDecorators:")
        print("  5. 🔺 Boost Priority")
        print("  6. 🔔 Add Reminder")
        print("  7. 🏷️  Add Tag")
        print("  8. 📎 Add Note")
        print("\nStrategies:")
        print("  9. Change Sort/Filter/Display")
        print("\nSingletons (NEW!):")
        print("  10. ⚙️  View/Edit Configuration")
        print("  11. 📊 View Global Statistics")
        print("  12. 📝 View Logs")
        print("\nHistory:")
        print(f"  13. Undo {'✅' if self.manager.command_history.can_undo() else '❌'}")
        print(f"  14. Redo {'✅' if self.manager.command_history.can_redo() else '❌'}")
        print("  15. Exit")
        print("=" * 75)

    def add_task_interactive(self):
        print("\n--- Add Task ---")
        title = input("Enter task title: ").strip()

        if not title:
            print("❌ Task title cannot be empty!")
            return

        description = input("Enter description (optional): ").strip()

        print("\nTask type:")
        print("1. Auto-detect")
        print("2. Normal")
        print("3. Urgent")
        print("4. Recurring")
        print("5. Important")

        type_choice = input("Choice (1-5, default 1): ").strip() or "1"

        task_type_map = {
            "1": "auto",
            "2": "normal",
            "3": "urgent",
            "4": "recurring",
            "5": "important",
        }

        task_type = task_type_map.get(type_choice, "auto")
        deadline = None
        frequency = 7

        if task_type == "urgent" or type_choice == "3":
            config = ConfigManager()
            default_hours = config.get("urgent_deadline_hours", 24)
            days_input = input(
                f"Days until deadline (default {default_hours/24:.0f}): "
            ).strip()
            days = int(days_input) if days_input.isdigit() else int(default_hours / 24)
            deadline = datetime.now() + timedelta(days=days)
        elif task_type == "recurring" or type_choice == "4":
            freq = input("Repeat every X days (default 7): ").strip()
            frequency = int(freq) if freq.isdigit() else 7

        if self.manager.add_task(title, description, task_type, deadline, frequency):
            print(f"✅ Task added!")
        else:
            print("❌ Failed to add task!")

    def boost_priority_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return

        self.manager.display_tasks()
        task_id = input("\nEnter task ID to boost priority: ").strip()
        levels = input("Boost by how many levels? (default 1): ").strip()
        levels = int(levels) if levels.isdigit() else 1

        if self.manager.decorate_task(task_id, "priority_boost", boost_levels=levels):
            print(f"✅ Priority boosted by {levels} level(s)!")
        else:
            print("❌ Task not found!")

    def add_reminder_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return

        self.manager.display_tasks()
        task_id = input("\nEnter task ID to add reminder: ").strip()
        hours = input("Remind in how many hours? (default 24): ").strip()
        hours = int(hours) if hours.isdigit() else 24

        reminder_time = datetime.now() + timedelta(hours=hours)

        if self.manager.decorate_task(task_id, "reminder", reminder_time=reminder_time):
            print(f"✅ Reminder set!")
        else:
            print("❌ Task not found!")

    def add_tag_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return

        self.manager.display_tasks()
        task_id = input("\nEnter task ID to tag: ").strip()
        tag = input("Enter tag (without #): ").strip()

        if tag and self.manager.decorate_task(task_id, "tag", tag=tag):
            print(f"✅ Tag #{tag} added!")
        else:
            print("❌ Failed to add tag!")

    def add_note_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return

        self.manager.display_tasks()
        task_id = input("\nEnter task ID to add note: ").strip()
        note = input("Enter note: ").strip()

        if note and self.manager.decorate_task(task_id, "notes", note=note):
            print(f"✅ Note added!")
        else:
            print("❌ Failed to add note!")

    def toggle_task_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return

        self.manager.display_tasks()
        task_id = input("\nEnter task ID to toggle: ").strip()

        if self.manager.toggle_task(task_id):
            print("✅ Task toggled!")
        else:
            print("❌ Task not found!")

    def delete_task_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return

        self.manager.display_tasks()
        task_id = input("\nEnter task ID to delete: ").strip()
        confirm = input("Are you sure? (y/n): ").strip().lower()

        if confirm == "y":
            if self.manager.delete_task(task_id):
                print("✅ Task deleted! (Can be undone)")
            else:
                print("❌ Task not found!")

    def change_strategies_menu(self):
        print("\n--- Change Strategies ---")
        print("1. Sort Strategy")
        print("2. Filter Strategy")
        print("3. Display Strategy")

        choice = input("Choose (1-3): ").strip()

        if choice == "1":
            self.change_sort_strategy()
        elif choice == "2":
            self.change_filter_strategy()
        elif choice == "3":
            self.change_display_strategy()

    def change_sort_strategy(self):
        print("\n--- Sort Strategies ---")
        for key, strategy in self.sort_strategies.items():
            current = (
                "← CURRENT"
                if strategy.name() == self.manager.sort_strategy.name()
                else ""
            )
            print(f"{key}. {strategy.name()} {current}")

        choice = input("\nChoose (1-2): ").strip()

        if choice in self.sort_strategies:
            self.manager.set_sort_strategy(self.sort_strategies[choice])
            print(f"✅ Sort: {self.sort_strategies[choice].name()}")

    def change_filter_strategy(self):
        print("\n--- Filter Strategies ---")
        for key, strategy in self.filter_strategies.items():
            current = (
                "← CURRENT"
                if strategy.name() == self.manager.filter_strategy.name()
                else ""
            )
            print(f"{key}. {strategy.name()} {current}")

        choice = input("\nChoose (1-3): ").strip()

        if choice in self.filter_strategies:
            self.manager.set_filter_strategy(self.filter_strategies[choice])
            print(f"✅ Filter: {self.filter_strategies[choice].name()}")

    def change_display_strategy(self):
        print("\n--- Display Strategies ---")
        for key, strategy in self.display_strategies.items():
            current = (
                "← CURRENT"
                if strategy.name() == self.manager.display_strategy.name()
                else ""
            )
            print(f"{key}. {strategy.name()} {current}")

        choice = input("\nChoose (1-2): ").strip()

        if choice in self.display_strategies:
            self.manager.set_display_strategy(self.display_strategies[choice])
            print(f"✅ Display: {self.display_strategies[choice].name()}")

    def view_config_interactive(self):
        """View and edit Singleton configuration"""
        config = ConfigManager()

        print("\n--- ⚙️  Configuration (Singleton) ---")
        print(f"Instance ID: {id(config)} (same every time!)")
        print("\nCurrent Settings:")

        all_config = config.get_all()
        for i, (key, value) in enumerate(all_config.items(), 1):
            print(f"  {i}. {key}: {value}")

        print("\nOptions:")
        print("  1. Edit a setting")
        print("  2. Reset to defaults")
        print("  3. Back")

        choice = input("\nChoose (1-3): ").strip()

        if choice == "1":
            key = input("Enter setting name: ").strip()
            if key in all_config:
                value = input(f"Enter new value for '{key}': ").strip()
                # Simple type conversion
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                config.set(key, value)
                print(f"✅ {key} = {value}")
            else:
                print("❌ Setting not found!")
        elif choice == "2":
            config.reset()
            print("✅ Configuration reset to defaults!")

    def view_global_stats_interactive(self):
        """View Singleton global statistics"""
        stats = GlobalStats()

        print("\n--- 📊 Global Statistics (Singleton) ---")
        print(f"Instance ID: {id(stats)} (same every time!)")
        print()

        all_stats = stats.get_all()
        for key, value in all_stats.items():
            if key == "app_start_time":
                print(f"{key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"{key}: {value}")

        print("\nOptions:")
        print("  1. Reset statistics")
        print("  2. Back")

        choice = input("\nChoose (1-2): ").strip()

        if choice == "1":
            stats.reset()
            print("✅ Statistics reset!")

    def view_logs_interactive(self):
        """View Singleton logger logs"""
        logger = TaskLogger()

        print("\n--- 📝 Task Logs (Singleton) ---")
        print(f"Instance ID: {id(logger)} (same every time!)")
        print()

        print("Filter by level:")
        print("  1. All")
        print("  2. INFO")
        print("  3. WARNING")
        print("  4. ERROR")

        level_map = {"1": None, "2": "INFO", "3": "WARNING", "4": "ERROR"}
        choice = input("\nChoose (1-4): ").strip()
        level = level_map.get(choice)

        logs = logger.get_logs(level=level, limit=20)

        if not logs:
            print("\nNo logs to display.")
            return

        print(f"\n--- Recent Logs (Last {len(logs)}) ---")
        for log in logs:
            timestamp = log["timestamp"].strftime("%H:%M:%S")
            level = log["level"]
            message = log["message"]
            print(f"[{timestamp}] {level}: {message}")

        print("\nOptions:")
        print("  1. Clear logs")
        print("  2. Back")

        choice = input("\nChoose (1-2): ").strip()

        if choice == "1":
            logger.clear()
            print("✅ Logs cleared!")

    def run(self):
        print("Welcome to Task Manager - Phase 7: Singleton Pattern!")
        print("🔒 Now featuring global singletons for config, logging, and stats!")
        print()

        # Demonstrate Singleton pattern
        config1 = ConfigManager()
        config2 = ConfigManager()
        print(f"🔍 Singleton Demo: config1 id={id(config1)}, config2 id={id(config2)}")
        print(f"   Same instance? {config1 is config2} ✅")

        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-15): ").strip()

            if choice == "1":
                self.add_task_interactive()
            elif choice == "2":
                self.manager.display_tasks()
            elif choice == "3":
                self.toggle_task_interactive()
            elif choice == "4":
                self.delete_task_interactive()
            elif choice == "5":
                self.boost_priority_interactive()
            elif choice == "6":
                self.add_reminder_interactive()
            elif choice == "7":
                self.add_tag_interactive()
            elif choice == "8":
                self.add_note_interactive()
            elif choice == "9":
                self.change_strategies_menu()
            elif choice == "10":
                self.view_config_interactive()
            elif choice == "11":
                self.view_global_stats_interactive()
            elif choice == "12":
                self.view_logs_interactive()
            elif choice == "13":
                if self.manager.undo():
                    print("✅ Last command undone!")
                else:
                    print("❌ Nothing to undo!")
            elif choice == "14":
                if self.manager.redo():
                    print("✅ Command redone!")
                else:
                    print("❌ Nothing to redo!")
            elif choice == "15":
                TaskLogger().info("Application shutting down")
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                print("Keep learning and building! 🚀")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-15.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()
