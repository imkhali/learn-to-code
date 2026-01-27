"""
Task Manager - Phase 5: Factory Pattern
Learn how to implement the Factory pattern for object creation

Patterns Implemented:
1. Observer Pattern - Event notifications
2. Strategy Pattern - Sorting/filtering/display
3. Command Pattern - Undo/redo
4. Factory Pattern - Task creation (NEW!)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Set
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum


# ============================================================================
# FACTORY PATTERN - TASK TYPES
# ============================================================================

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5  # For boosted priority


class Task(ABC):
    """Abstract base class for all task types"""
    
    def __init__(self, title: str, description: str = ""):
        if not title.strip():
            raise ValueError("Task title cannot be empty")
        
        self.id = self._generate_id()
        self.title = title.strip()
        self.description = description.strip()
        self.completed = False
        self.created_at = datetime.now()

    def is_type(self, task_class) -> bool:
        """Check if this task is of given type"""
        return isinstance(self, task_class)
    
    def has_decorator(self, decorator_class) -> bool:
        """Check if this task has a specific decorator"""
        return isinstance(self, decorator_class)
    
    def get_base_task(self) -> 'Task':
        """Get the innermost unwrapped task"""
        return self
    
    @staticmethod
    def _generate_id() -> str:
        return str(int(datetime.now().timestamp() * 1000000))
    
    @abstractmethod
    def get_priority(self) -> TaskPriority:
        """Return the priority of this task"""
        pass
    
    @abstractmethod
    def get_type_name(self) -> str:
        """Return the type name for display"""
        pass
    
    @abstractmethod
    def get_display_icon(self) -> str:
        """Return an icon representing this task type"""
        pass
    
    def toggle_completion(self):
        self.completed = not self.completed
    
    def get_full_description(self) -> str:
        """Get complete description (base implementation)"""
        return self.description
    
    def get_tags(self) -> Set[str]:
        """Get tags (base implementation - no tags)"""
        return set()
    
    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        icon = self.get_display_icon()
        return f"{status} {icon} {self.title}"


class NormalTask(Task):
    """Standard task with normal priority"""
    
    def get_priority(self) -> TaskPriority:
        return TaskPriority.NORMAL
    
    def get_type_name(self) -> str:
        return "Normal"
    
    def get_display_icon(self) -> str:
        return "📝"


class UrgentTask(Task):
    """High-priority task that needs immediate attention"""

    DEFAULT_DEADLINE_HOURS = 24
    
    def __init__(self, title: str, description: str = "", deadline: Optional[datetime] = None):
        super().__init__(title, description)
        
        if deadline is None:
            self.deadline = datetime.now() + timedelta(hours=self.DEFAULT_DEADLINE_HOURS)
        else:
            self.deadline = deadline
    
    @classmethod
    def set_default_deadline_hours(cls, hours: int):
        """Allow users to configure default deadline"""
        cls.DEFAULT_DEADLINE_HOURS = hours
    
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
        if self.is_overdue():
            return f"{base} ⚠️ OVERDUE"
        return f"{base} (Due: {self.deadline.strftime('%m/%d %H:%M')})"


class RecurringTask(Task):
    """Task that repeats on a schedule"""
    
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
        """When completed, schedule next occurrence"""
        if not self.completed:
            self.completed = True
            self.next_occurrence = datetime.now() + timedelta(days=self.frequency_days)
        else:
            self.completed = False
    
    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (Every {self.frequency_days} days)"


class ImportantTask(Task):
    """High-priority task but not time-sensitive"""
    
    def get_priority(self) -> TaskPriority:
        return TaskPriority.HIGH
    
    def get_type_name(self) -> str:
        return "Important"
    
    def get_display_icon(self) -> str:
        return "⭐"


# ============================================================================
# DECORATOR PATTERN - TASK DECORATORS
# ============================================================================

class TaskDecorator(Task):
    """
    Base decorator class.
    
    This is the DECORATOR PATTERN in action!
    - Wraps a Task object
    - Implements the same interface (Task)
    - Delegates to the wrapped task
    - Can add additional behavior
    """

    def __init__(self, task: Task):
        # Don't call super().__init__() - we're wrapping, not creating!
        self._wrapped_task = task
    
    def is_type(self, task_class) -> bool:
        if isinstance(self, task_class):
            return True
        return self._wrapped_task.is_type(task_class)
    
    def has_decorator(self, decorator_class) -> bool:
        if isinstance(self, decorator_class):
            return True
        return self._wrapped_task.has_decorator(decorator_class)
    
    def get_base_task(self) -> Task:
        """Unwrap all decorators"""
        return self._wrapped_task.get_base_task()

    # Delegate all Task interface methods to wrapped task
    @property
    def id(self):
        return self._wrapped_task.id
    
    @property
    def title(self):
        return self._wrapped_task.title
    
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
    
    def __str__(self) -> str:
        return str(self._wrapped_task)
    

class PriorityBoostDecorator(TaskDecorator):
    """
    Decorator that boosts task priority.
    Useful for temporarily elevating a task's importance.
    """

    def __init__(self, task: Task, boost_levels: int = 1):
        super().__init__(task)
        self.boost_levels = boost_levels
    
    def get_priority(self) -> TaskPriority:
        """Boost priority by specified levels"""
        base_priority = self._wrapped_task.get_priority()
        boosted_value = min(base_priority.value + self.boost_levels, TaskPriority.CRITICAL.value)
        return TaskPriority(boosted_value)
    
    def get_display_icon(self) -> str:
        """Add boost indicator to icon"""
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}🔺"  # Up arrow indicates boost
    
    def __str__(self) -> str:
        base = str(self._wrapped_task)
        return f"{base} [PRIORITY BOOSTED]"

    
class ReminderDecorator(TaskDecorator):
    """Decorator that adds reminder functionality to any task."""

    def __init__(self, task: Task, reminder_time: datetime):
        super().__init__(task)
        self.reminder_time = reminder_time

    def is_reminder_due(self) -> bool:
        return datetime.now() >= self.reminder_time and not self.completed
        
    def get_display_icon(self) -> str:
        """Add bell icon to indicate reminder"""
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}🔔"
    
    def __str__(self) -> str:
        base = str(self._wrapped_task)
        if self.is_reminder_due():
            return f"{base} 🔔 REMINDER!"
        return f"{base} (Reminder: {self.reminder_time.strftime('%m/%d %H:%M')})"
    

class TagDecorator(TaskDecorator):
    """
    Decorator that adds tags/labels to tasks.
    Multiple TagDecorators can be stacked!
    """
    
    def __init__(self, task: Task, tag: str):
        super().__init__(task)
        self.tag = tag
    
    def get_tags(self) -> Set[str]:
        """Add this tag to the wrapped task's tags"""
        tags = self._wrapped_task.get_tags()
        tags.add(self.tag)
        return tags
    
    def get_display_icon(self) -> str:
        """Add tag icon"""
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
    """
    Decorator that adds additional notes/comments to tasks.
    """

    def __init__(self, task: Task, note: str):
        super().__init__(task)
        self.note = note
    
    def get_full_description(self) -> str:
        """Append note to description"""
        base_desc = self._wrapped_task.get_full_description()
        if base_desc:
            return f"{base_desc}\n\nNote: {self.note}"
        return f"Note: {self.note}"

    def get_display_icon(self) -> str:
        """Add note icon"""
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}📎"
    
    def __str__(self) -> str:
        base = str(self._wrapped_task)
        return f"{base} 📎"

class DelegatedDecorator(TaskDecorator):
    """Decorator that marks a task as delegated to someone."""

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
# FACTORY PATTERN - TASK FACTORY
# ============================================================================

class TaskFactory:
    """
    Factory class that creates appropriate task types.
    
    This is the FACTORY PATTERN in action!
    - Client doesn't need to know about NormalTask, UrgentTask, etc.
    - Factory encapsulates the creation logic
    - Easy to add new task types without changing client code
    """
    
    # Keywords that trigger specific task types
    URGENT_KEYWORDS = ['urgent', 'asap', 'critical', 'emergency', 'immediately']
    RECURRING_KEYWORDS = ['daily', 'weekly', 'monthly', 'every', 'recurring']
    IMPORTANT_KEYWORDS = ['important', 'priority', 'key', 'crucial', 'vital']
    
    @staticmethod
    def create_task(title: str, description: str = "", task_type: str = "auto", 
                   deadline: Optional[datetime] = None, frequency_days: int = 7) -> Task:
        """
        Factory method to create tasks.
        
        Args:
            title: Task title
            description: Task description
            task_type: "auto", "normal", "urgent", "recurring", "important"
            deadline: Deadline for urgent tasks
            frequency_days: Frequency for recurring tasks
        
        Returns:
            Appropriate Task subclass instance
        """
        
        # AUTO mode: Analyze title/description to determine type
        if task_type == "auto":
            task_type = TaskFactory._detect_task_type(title, description)
        
        # Create appropriate task type
        if task_type == "urgent":
            return UrgentTask(title, description, deadline)
        elif task_type == "recurring":
            return RecurringTask(title, description, frequency_days)
        elif task_type == "important":
            return ImportantTask(title, description)
        else:  # "normal" or unknown
            return NormalTask(title, description)
    
    @staticmethod
    def _detect_task_type(title: str, description: str) -> str:
        """
        Smart detection: Analyze text to determine task type.
        This is the "intelligence" of the factory!
        """
        text = (title + " " + description).lower()
        
        # Check for urgent keywords
        if any(keyword in text for keyword in TaskFactory.URGENT_KEYWORDS):
            return "urgent"
        
        # Check for recurring keywords
        if any(keyword in text for keyword in TaskFactory.RECURRING_KEYWORDS):
            return "recurring"
        
        # Check for important keywords
        if any(keyword in text for keyword in TaskFactory.IMPORTANT_KEYWORDS):
            return "important"
        
        return "normal"
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Return list of available task types"""
        return ["auto", "normal", "urgent", "recurring", "important"]


# ============================================================================
# OBSERVER PATTERN
# ============================================================================

class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: dict):
        pass


class NotificationObserver(Observer):
    def update(self, event: str, data: dict):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Build notification message based on event type
        if event == "task_added":
            task_type = data.get('type', 'task')
            message = f"🔔 [{timestamp}] New {task_type} added: '{data['title']}'"
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
            return  # Unknown event, don't print anything
        
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
# STRATEGY PATTERN - SORTING
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


class SortByTypeStrategy(SortStrategy):
    def sort(self, tasks: List[Task]) -> List[Task]:
        return sorted(tasks, key=lambda t: t.get_type_name())
    
    def name(self) -> str:
        return "Type"


class SortByDateStrategy(SortStrategy):
    def sort(self, tasks: List[Task]) -> List[Task]:
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def name(self) -> str:
        return "Date (Newest First)"


class SortByTitleStrategy(SortStrategy):
    def sort(self, tasks: List[Task]) -> List[Task]:
        return sorted(tasks, key=lambda t: t.title.lower())
    
    def name(self) -> str:
        return "Title (A-Z)"


# ============================================================================
# STRATEGY PATTERN - FILTERING
# ============================================================================

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


class ShowRecurringFilter(FilterStrategy):

    def filter(self, tasks: List[Task]) -> List[Task]:
        return [t for t in tasks if t.is_type(RecurringTask)]
    
    def name(self) -> str:
        return "Recurring Only"




class ShowOverdueFilter(FilterStrategy):
    def filter(self, tasks: List[Task]) -> List[Task]:
        return [t for t in tasks if isinstance(t, UrgentTask) and t.is_overdue()]
    
    def name(self) -> str:
        return "Overdue Only"


# ============================================================================
# STRATEGY PATTERN - DISPLAY
# ============================================================================

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
        
        for i, task in enumerate(tasks, 1):
            status = "✓ DONE" if task.completed else "○ PENDING"
            print(f"\n{i}. [{task.id}] {status}")
            print(f"   Type: {task.get_type_name()} {task.get_display_icon()}")
            print(f"   Priority: {task.get_priority().name}")
            print(f"   Title: {task.title}")
            if task.description:
                print(f"   Description: {task.description}")
            print(f"   Created: {task.created_at.strftime('%Y-%m-%d %H:%M')}")
            
            if isinstance(task, UrgentTask):
                print(f"   Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}")
                if task.is_overdue():
                    print(f"   ⚠️ OVERDUE!")
            elif isinstance(task, RecurringTask):
                print(f"   Frequency: Every {task.frequency_days} days")
                print(f"   Next: {task.next_occurrence.strftime('%Y-%m-%d')}")
    
    def name(self) -> str:
        return "Detailed"


# ============================================================================
# COMMAND PATTERN
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
    def __init__(self, task_manager: 'TaskManager', title: str, description: str = "",
                 task_type: str = "auto", deadline: Optional[datetime] = None, 
                 frequency_days: int = 7):
        self.task_manager = task_manager
        self.title = title
        self.description = description
        self.task_type = task_type
        self.deadline = deadline
        self.frequency_days = frequency_days
        self.task = None
    
    def execute(self):
        self.task = self.task_manager._add_task_internal(
            self.title, self.description, self.task_type, 
            self.deadline, self.frequency_days
        )
        self.task_manager.notify("command_executed", {
            'command': self.get_description()
        })
    
    def undo(self):
        if self.task:
            self.task_manager._delete_task_internal(self.task.id)
            self.task_manager.notify("command_undone", {
                'command': self.get_description()
            })
    
    def get_description(self) -> str:
        type_name = self.task.get_type_name() if self.task else "Task"
        return f"Add {type_name}: '{self.title}'"


class DeleteTaskCommand(Command):
    def __init__(self, task_manager: 'TaskManager', task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id
        self.deleted_task = None
    
    def execute(self):
        task = self.task_manager.get_task_by_id(self.task_id)
        if task:
            self.deleted_task = deepcopy(task)
            self.task_manager._delete_task_internal(self.task_id)
            self.task_manager.notify("command_executed", {
                'command': self.get_description()
            })
    
    def undo(self):
        if self.deleted_task:
            self.task_manager.tasks.append(self.deleted_task)
            self.task_manager.notify("command_undone", {
                'command': self.get_description()
            })
    
    def get_description(self) -> str:
        title = self.deleted_task.title if self.deleted_task else "Unknown"
        return f"Delete Task: '{title}'"


class ToggleTaskCommand(Command):
    def __init__(self, task_manager: 'TaskManager', task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id
        self.task = None
    
    def execute(self):
        self.task = self.task_manager.get_task_by_id(self.task_id)
        if self.task:
            self.task.toggle_completion()
            
            event = "task_completed" if self.task.completed else "task_uncompleted"
            self.task_manager.notify(event, {
                'id': self.task.id,
                'title': self.task.title
            })
            self.task_manager.notify("command_executed", {
                'command': self.get_description()
            })
    
    def undo(self):
        if self.task:
            self.task.toggle_completion()
            
            event = "task_completed" if self.task.completed else "task_uncompleted"
            self.task_manager.notify(event, {
                'id': self.task.id,
                'title': self.task.title
            })
            self.task_manager.notify("command_undone", {
                'command': self.get_description()
            })
    
    def get_description(self) -> str:
        if self.task:
            status = "Complete" if not self.task.completed else "Uncomplete"
            return f"{status} Task: '{self.task.title}'"
        return "Toggle Task"


class DecorateTaskCommand(Command):
    """NEW: Command to decorate a task"""
    def __init__(self, task_manager: 'TaskManager', task_id: str, decorator_type: str, **kwargs):
        self.task_manager = task_manager
        self.task_id = task_id
        self.decorator_type = decorator_type
        self.kwargs = kwargs
        self.original_task_index = None
    
    
    def execute(self):
        task = self.task_manager.get_task_by_id(self.task_id)
        if not task:
            return
    
        

class CommandHistory:
    def __init__(self):
        self.history: List[Command] = []
        self.undo_stack: List[Command] = []
    
    def execute_command(self, command: Command):
        command.execute()
        self.history.append(command)
        self.undo_stack.clear()
    
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
        return True
    
    def can_undo(self) -> bool:
        return len(self.history) > 0
    
    def can_redo(self) -> bool:
        return len(self.undo_stack) > 0


# ============================================================================
# TASK MANAGER
# ============================================================================

class TaskManager(Subject):
    def __init__(self):
        super().__init__()
        self.tasks: List[Task] = []
        self.command_history = CommandHistory()
        self.factory = TaskFactory()
        
        self.sort_strategy: SortStrategy = SortByPriorityStrategy()
        self.filter_strategy: FilterStrategy = ShowAllFilter()
        self.display_strategy: DisplayStrategy = SimpleDisplayStrategy()
    
    def set_sort_strategy(self, strategy: SortStrategy):
        self.sort_strategy = strategy
    
    def set_filter_strategy(self, strategy: FilterStrategy):
        self.filter_strategy = strategy
    
    def set_display_strategy(self, strategy: DisplayStrategy):
        self.display_strategy = strategy
    
    def add_task(self, title: str, description: str = "", task_type: str = "auto",
                deadline: Optional[datetime] = None, frequency_days: int = 7) -> bool:
        if not title.strip():
            return False
        
        command = AddTaskCommand(self, title, description, task_type, deadline, frequency_days)
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
    
    def undo(self) -> bool:
        return self.command_history.undo()
    
    def redo(self) -> bool:
        return self.command_history.redo()
    
    def _add_task_internal(self, title: str, description: str, task_type: str,
                          deadline: Optional[datetime], frequency_days: int) -> Task:
        """Uses Factory to create task!"""
        task = self.factory.create_task(title, description, task_type, deadline, frequency_days)
        self.tasks.append(task)
        self.notify("task_added", {
            'id': task.id,
            'title': task.title,
            'type': task.get_type_name(),
            'description': task.description
        })
        return task
    
    def _delete_task_internal(self, task_id: str) -> bool:
        task = self.get_task_by_id(task_id)
        if task:
            self.notify("task_deleted", {'id': task.id, 'title': task.title})
            self.tasks.remove(task)
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
        print(f"\n--- Tasks ({self.filter_strategy.name()}, {self.sort_strategy.name()}) ---")
        self.display_strategy.display(tasks)
    
    def get_stats(self) -> dict:
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.completed)
        pending = total - completed
        
        urgent_count = sum(1 for t in self.tasks if isinstance(t, UrgentTask))
        recurring_count = sum(1 for t in self.tasks if isinstance(t, RecurringTask))
        important_count = sum(1 for t in self.tasks if isinstance(t, ImportantTask))
        normal_count = sum(1 for t in self.tasks if isinstance(t, NormalTask))
        overdue_count = sum(1 for t in self.tasks if isinstance(t, UrgentTask) and t.is_overdue())
        
        return {
            'total': total,
            'completed': completed,
            'pending': pending,
            'completion_rate': (completed / total * 100) if total > 0 else 0,
            'urgent': urgent_count,
            'recurring': recurring_count,
            'important': important_count,
            'normal': normal_count,
            'overdue': overdue_count
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

class TaskManagerCLI:
    def __init__(self):
        self.manager = TaskManager()
        
        notification_observer = NotificationObserver()
        self.manager.attach(notification_observer)
        
        self.sort_strategies = {
            '1': SortByPriorityStrategy(),
            '2': SortByTypeStrategy(),
            '3': SortByDateStrategy(),
            '4': SortByTitleStrategy()
        }
        
        self.filter_strategies = {
            '1': ShowAllFilter(),
            '2': ShowUrgentFilter(),
            '3': ShowRecurringFilter(),
            '4': ShowPendingFilter(),
            '5': ShowOverdueFilter()
        }
        
        self.display_strategies = {
            '1': SimpleDisplayStrategy(),
            '2': DetailedDisplayStrategy()
        }
        
        self.running = True
    
    def display_menu(self):
        print("\n" + "="*70)
        print("📋 TASK MANAGER - Phase 5: Factory Pattern")
        print("="*70)
        print("Tasks:")
        print("  1. Add Task (Smart Detection)")
        print("  2. Add Task (Manual Type Selection)")
        print("  3. View Tasks")
        print("  4. Complete/Uncomplete Task")
        print("  5. Delete Task")
        print("\nStrategies:")
        print("  6. Change Sort Strategy")
        print("  7. Change Filter Strategy")
        print("  8. Change Display Strategy")
        print("\nHistory:")
        print(f"  9. Undo {'✅' if self.manager.command_history.can_undo() else '❌'}")
        print(f"  10. Redo {'✅' if self.manager.command_history.can_redo() else '❌'}")
        print("\nInfo:")
        print("  11. View Statistics")
        print("  12. Exit")
        print("="*70)
    
    def add_task_auto(self):
        print("\n--- Add Task (Smart Detection) ---")
        print("💡 Tip: Use words like 'urgent', 'weekly', 'important' for auto-detection")
        title = input("Enter task title: ").strip()
        
        if not title:
            print("❌ Task title cannot be empty!")
            return
        
        description = input("Enter description (optional): ").strip()
        
        if self.manager.add_task(title, description, task_type="auto"):
            print(f"✅ Task added! Factory detected the best type")
        else:
            print("❌ Failed to add task!")
    
    def add_task_manual(self):
        print("\n--- Add Task (Manual Type) ---")
        title = input("Enter task title: ").strip()
        
        if not title:
            print("❌ Task title cannot be empty!")
            return
        
        description = input("Enter description (optional): ").strip()
        
        print("\nSelect task type:")
        print("1. Normal 📝")
        print("2. Urgent 🔥 (has deadline)")
        print("3. Recurring 🔄 (repeats)")
        print("4. Important ⭐ (high priority)")
        
        type_choice = input("Choice (1-4): ").strip()
        
        task_type = "normal"
        deadline = None
        frequency = 7
        
        if type_choice == "2":
            task_type = "urgent"
            days = input("Days until deadline (default 1): ").strip()
            days = int(days) if days.isdigit() else 1
            deadline = datetime.now() + timedelta(days=days)
        elif type_choice == "3":
            task_type = "recurring"
            freq = input("Repeat every X days (default 7): ").strip()
            frequency = int(freq) if freq.isdigit() else 7
        elif type_choice == "4":
            task_type = "important"
        
        if self.manager.add_task(title, description, task_type, deadline, frequency):
            print(f"✅ {task_type.capitalize()} task added!")
        else:
            print("❌ Failed to add task!")
    
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
        
        if confirm == 'y':
            if self.manager.delete_task(task_id):
                print("✅ Task deleted! (Can be undone)")
            else:
                print("❌ Task not found!")
    
    def change_sort_strategy(self):
        print("\n--- Sort Strategies ---")
        for key, strategy in self.sort_strategies.items():
            current = "← CURRENT" if strategy.name() == self.manager.sort_strategy.name() else ""
            print(f"{key}. {strategy.name()} {current}")
        
        choice = input("\nChoose (1-4): ").strip()
        
        if choice in self.sort_strategies:
            self.manager.set_sort_strategy(self.sort_strategies[choice])
            print(f"✅ Sort: {self.sort_strategies[choice].name()}")
        else:
            print("❌ Invalid choice!")
    
    def change_filter_strategy(self):
        print("\n--- Filter Strategies ---")
        for key, strategy in self.filter_strategies.items():
            current = "← CURRENT" if strategy.name() == self.manager.filter_strategy.name() else ""
            print(f"{key}. {strategy.name()} {current}")
        
        choice = input("\nChoose (1-5): ").strip()
        
        if choice in self.filter_strategies:
            self.manager.set_filter_strategy(self.filter_strategies[choice])
            print(f"✅ Filter: {self.filter_strategies[choice].name()}")
        else:
            print("❌ Invalid choice!")
    
    def change_display_strategy(self):
        print("\n--- Display Strategies ---")
        for key, strategy in self.display_strategies.items():
            current = "← CURRENT" if strategy.name() == self.manager.display_strategy.name() else ""
            print(f"{key}. {strategy.name()} {current}")
        
        choice = input("\nChoose (1-2): ").strip()
        
        if choice in self.display_strategies:
            self.manager.set_display_strategy(self.display_strategies[choice])
            print(f"✅ Display: {self.display_strategies[choice].name()}")
        else:
            print("❌ Invalid choice!")
    
    def view_statistics(self):
        stats = self.manager.get_stats()
        print("\n--- 📊 Task Statistics ---")
        print(f"Total Tasks: {stats['total']}")
        print(f"Completed: {stats['completed']}")
        print(f"Pending: {stats['pending']}")
        print(f"Completion Rate: {stats['completion_rate']:.1f}%")
        print("\n--- Task Types ---")
        print(f"📝 Normal: {stats['normal']}")
        print(f"🔥 Urgent: {stats['urgent']}")
        print(f"⭐ Important: {stats['important']}")
        print(f"🔄 Recurring: {stats['recurring']}")
        if stats['overdue'] > 0:
            print(f"\n⚠️  Overdue: {stats['overdue']}")
    
    def run(self):
        print("Welcome to Task Manager - Phase 5: Factory Pattern!")
        print("🏭 Now featuring smart task type detection and creation")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-12): ").strip()
            
            if choice == '1':
                self.add_task_auto()
            elif choice == '2':
                self.add_task_manual()
            elif choice == '3':
                self.manager.display_tasks()
            elif choice == '4':
                self.toggle_task_interactive()
            elif choice == '5':
                self.delete_task_interactive()
            elif choice == '6':
                self.change_sort_strategy()
            elif choice == '7':
                self.change_filter_strategy()
            elif choice == '8':
                self.change_display_strategy()
            elif choice == '9':
                if self.manager.undo():
                    print("✅ Last command undone!")
                else:
                    print("❌ Nothing to undo!")
            elif choice == '10':
                if self.manager.redo():
                    print("✅ Command redone!")
                else:
                    print("❌ Nothing to redo!")
            elif choice == '11':
                self.view_statistics()
            elif choice == '12':
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                print("Keep learning and building! 🚀")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-12.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()