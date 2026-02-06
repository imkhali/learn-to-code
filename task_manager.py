"""
Task Manager - Phase 9: State Pattern
Learn how to implement the State pattern for state-dependent behavior

Patterns Implemented:
1. Observer Pattern - Event notifications
2. Strategy Pattern - Sorting/filtering/display  
3. Command Pattern - Undo/redo functionality
4. Factory Pattern - Object creation by type
5. Decorator Pattern - Add features dynamically
6. Singleton Pattern - Single instance management
7. Builder Pattern - Fluent object construction
8. State Pattern - Task workflow states (NEW!)
"""

from datetime import datetime, timedelta
from typing import List, Optional, Set, Dict, Any
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import threading

# ============================================================================
# STATE PATTERN - TASK STATES
# ============================================================================

class TaskState(ABC):
    """
    Abstract base class for task states.
    
    This is the STATE PATTERN in action!
    - Each state is a separate class
    - State-specific behavior encapsulated
    - Clean state transitions
    - No giant if-elif chains
    """
    
    @abstractmethod
    def get_state_name(self) -> str:
        """Return the name of this state"""
        pass
    
    @abstractmethod
    def get_state_icon(self) -> str:
        """Return an icon representing this state"""
        pass
    
    @abstractmethod
    def can_transition_to(self, new_state: str) -> bool:
        """Check if transition to new state is allowed"""
        pass
    
    @abstractmethod
    def on_enter(self, task: 'StatefulTask'):
        """Called when entering this state"""
        pass
    
    @abstractmethod
    def on_exit(self, task: 'StatefulTask'):
        """Called when exiting this state"""
        pass
    
    def can_edit(self) -> bool:
        """Can task be edited in this state? Default: yes"""
        return True
    
    def can_delete(self) -> bool:
        """Can task be deleted in this state? Default: yes"""
        return True
    
    def can_complete(self) -> bool:
        """Can task be marked complete in this state? Default: yes"""
        return True


class DraftState(TaskState):
    """Task is in draft - not yet active"""
    
    def get_state_name(self) -> str:
        return "Draft"
    
    def get_state_icon(self) -> str:
        return "📝"
    
    def can_transition_to(self, new_state: str) -> bool:
        # From Draft, can go to: Active, Archived
        return new_state in ["Active", "Archived"]
    
    def on_enter(self, task: 'StatefulTask'):
        TaskLogger().info(f"Task '{task.title}' entered Draft state")
    
    def on_exit(self, task: 'StatefulTask'):
        TaskLogger().info(f"Task '{task.title}' left Draft state")
    
    def can_complete(self) -> bool:
        return False  # Can't complete a draft


class ActiveState(TaskState):
    """Task is active and in progress"""
    
    def get_state_name(self) -> str:
        return "Active"
    
    def get_state_icon(self) -> str:
        return "🚀"
    
    def can_transition_to(self, new_state: str) -> bool:
        # From Active, can go to: In Review, Completed, On Hold, Archived
        return new_state in ["In Review", "Completed", "On Hold", "Archived"]
    
    def on_enter(self, task: 'StatefulTask'):
        task.started_at = datetime.now()
        TaskLogger().info(f"Task '{task.title}' is now Active")
    
    def on_exit(self, task: 'StatefulTask'):
        pass


class InReviewState(TaskState):
    """Task is under review"""
    
    def get_state_name(self) -> str:
        return "In Review"
    
    def get_state_icon(self) -> str:
        return "👀"
    
    def can_transition_to(self, new_state: str) -> bool:
        # From Review, can go to: Active (needs changes), Completed, Archived
        return new_state in ["Active", "Completed", "Archived"]
    
    def on_enter(self, task: 'StatefulTask'):
        TaskLogger().info(f"Task '{task.title}' is In Review")
    
    def on_exit(self, task: 'StatefulTask'):
        pass
    
    def can_edit(self) -> bool:
        return False  # Can't edit while in review


class CompletedState(TaskState):
    """Task is completed"""
    
    def get_state_name(self) -> str:
        return "Completed"
    
    def get_state_icon(self) -> str:
        return "✅"
    
    def can_transition_to(self, new_state: str) -> bool:
        # From Completed, can go to: Active (reopen), Archived
        return new_state in ["Active", "Archived"]
    
    def on_enter(self, task: 'StatefulTask'):
        task.completed = True
        task.completed_at = datetime.now()
        TaskLogger().info(f"Task '{task.title}' completed!")
        GlobalStats().increment('tasks_completed')
    
    def on_exit(self, task: 'StatefulTask'):
        task.completed = False
        task.completed_at = None
    
    def can_edit(self) -> bool:
        return False  # Can't edit completed tasks
    
    def can_complete(self) -> bool:
        return False  # Already completed


class OnHoldState(TaskState):
    """Task is temporarily on hold"""
    
    def get_state_name(self) -> str:
        return "On Hold"
    
    def get_state_icon(self) -> str:
        return "⏸️"
    
    def can_transition_to(self, new_state: str) -> bool:
        # From On Hold, can go to: Active, Archived
        return new_state in ["Active", "Archived"]
    
    def on_enter(self, task: 'StatefulTask'):
        TaskLogger().info(f"Task '{task.title}' is On Hold")
    
    def on_exit(self, task: 'StatefulTask'):
        pass
    
    def can_complete(self) -> bool:
        return False  # Can't complete while on hold


class ArchivedState(TaskState):
    """Task is archived (no longer active)"""
    
    def get_state_name(self) -> str:
        return "Archived"
    
    def get_state_icon(self) -> str:
        return "📦"
    
    def can_transition_to(self, new_state: str) -> bool:
        # From Archived, can only restore to Draft
        return new_state == "Draft"
    
    def on_enter(self, task: 'StatefulTask'):
        task.archived_at = datetime.now()
        TaskLogger().info(f"Task '{task.title}' archived")
    
    def on_exit(self, task: 'StatefulTask'):
        task.archived_at = None
    
    def can_edit(self) -> bool:
        return False
    
    def can_delete(self) -> bool:
        return True
    
    def can_complete(self) -> bool:
        return False


# State Registry (for easy access)
TASK_STATES = {
    "Draft": DraftState(),
    "Active": ActiveState(),
    "In Review": InReviewState(),
    "Completed": CompletedState(),
    "On Hold": OnHoldState(),
    "Archived": ArchivedState()
}


# ============================================================================
# STATEFUL TASK (Task with State Pattern)
# ============================================================================

class StatefulTask:
    """
    A task that uses the State Pattern for workflow management.
    
    State transitions:
    Draft → Active → In Review → Completed → Archived
              ↓         ↓          ↓
          On Hold    Active     Active
    """
    
    def __init__(self, title: str, description: str = ""):
        if not title.strip():
            raise ValueError("Task title cannot be empty")
        
        self.id = self._generate_id()
        self.title = title.strip()
        self.description = description.strip()
        self.created_at = datetime.now()
        
        # State-specific attributes
        self.state: TaskState = TASK_STATES["Draft"]  # Start in Draft
        self.completed = False
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.archived_at: Optional[datetime] = None
        
        # Tags and notes
        self.tags: Set[str] = set()
        self.notes: List[str] = []
        
        TaskLogger().info(f"Stateful task created: {self.title}")
        GlobalStats().increment('tasks_created')
    
    @staticmethod
    def _generate_id() -> str:
        return str(int(datetime.now().timestamp() * 1000000))
    
    # ========== STATE TRANSITIONS ==========
    
    def transition_to(self, new_state_name: str) -> bool:
        """
        Transition to a new state.
        Returns True if successful, False if invalid transition.
        """
        # Check if transition is allowed
        if not self.state.can_transition_to(new_state_name):
            TaskLogger().warning(
                f"Invalid transition: {self.state.get_state_name()} → {new_state_name}",
                task_id=self.id
            )
            return False
        
        # Get new state
        new_state = TASK_STATES.get(new_state_name)
        if not new_state:
            return False

        # Perform the transition
        old_state_name = self.state.get_state_name()

        self.state.on_exit(self)
        self.state = new_state
        self.state.on_enter(self)

        TaskLogger().info(
            f"State transition: {old_state_name} -> {new_state_name}",
            task_id=self.id,
            title=self.title
        )
    
    def start(self) -> bool:
        """Start the task (Draft → Active)"""
        return self.transition_to("Active")
    
    def send_to_review(self) -> bool:
        return self.transition_to("In Review")
    
    def complete_task(self) -> bool:
        """Complete the task (any state → Completed)"""
        return self.transition_to("Completed")
    
    def hold(self) -> bool:
        """Put on hold (Active → On Hold)"""
        return self.transition_to("On Hold")
    
    def resume(self) -> bool:
        """Resume from hold (On Hold → Active)"""
        return self.transition_to("Active")
    
    def reopen(self) -> bool:
        """Reopen completed task (Completed → Active)"""
        return self.transition_to("Active")
    
    def archive(self) -> bool:
        """Archive the task (any state → Archived)"""
        return self.transition_to("Archived")
    
    def restore(self) -> bool:
        """Restore from archive (Archived → Draft)"""
        return self.transition_to("Draft")
    
    # ========== STATE-DEPENDENT BEHAVIOR ==========
    
    def can_edit(self) -> bool:
        """Check if task can be edited in current state"""
        return self.state.can_edit()
    
    def can_delete(self) -> bool:
        return self.state.can_delete()

    def can_complete(self) -> bool:
        """Check if task can be completed in current state"""
        return self.state.can_complete()
    
    def update_title(self, new_title: str) -> bool:
        """Update title (only if editable)"""
        if not self.can_edit():
            print(f"❌ Cannot edit task in {self.state.get_state_name()} state")
            return False
        
        self.title = new_title.strip()
        return True
    
    def update_description(self, new_description: str) -> bool:
        """Update description (only if editable)"""
        if not self.can_edit():
            print(f"❌ Cannot edit task in {self.state.get_state_name()} state")
            return False
        
        self.description = new_description.strip()
        return True
    
    def add_tag(self, tag: str):
        """Add a tag"""
        self.tags.add(tag)
    
    def add_note(self, note: str):
        """Add a note"""
        self.notes.append(note)

    def get_state_summary(self) -> str:
        """Get a summary of current state"""
        icon = self.state.get_state_icon()
        name = self.state.get_state_name()

        info = []
        if self.started_at:
            info.append(f"Started: {self.started_at.strftime('%m/%d %H:%M')}")
        if self.completed_at:
            info.append(f"Completed: {self.completed_at.strftime('%m/%d %H:%M')}")
        if self.archived_at:
            info.append(f"Archived: {self.archived_at.strftime('%m/%d %H:%M')}")
        
        info_str = " | ".join(info) if info else ""
        return f"{icon} {name}" + (f" ({info_str})" if info_str else "")
        
    def __str__(self) -> str:
        state_icon = self.state.get_state_icon()
        state_name = self.state.get_state_name()
        tags_str = f" [{''.join(f'#{t} ' for t in sorted(self.tags))}]" if self.tags else ""
        return f"{state_icon} [{state_name}] {self.title}{tags_str}"


# ============================================================================
# BUILDER PATTERN - TASK BUILDER
# ============================================================================

class TaskBuilder:
    """
    Builder for creating complex tasks with fluent interface.
    
    This is the BUILDER PATTERN in action!
    - Constructs objects step by step
    - Fluent interface (method chaining)
    - Validation before building
    - Readable and maintainable
    
    Example:
        task = (TaskBuilder()
                .with_title("Fix critical bug")
                .with_description("Production issue affecting users")
                .make_urgent(deadline_hours=4)
                .add_tag("critical")
                .add_tag("backend")
                .delegate_to("Alice")
                .build())
    """
    
    def __init__(self):
        # Required fields
        self._title: Optional[str] = None
        
        # Optional fields
        self._description: str = ""
        self._task_type: str = "normal"
        self._deadline: Optional[datetime] = None
        self._frequency_days: int = 7
        
        # Decorators to apply
        self._tags: Set[str] = set()
        self._notes: List[str] = []
        self._priority_boost: int = 0
        self._reminder_time: Optional[datetime] = None
        self._delegated_to: Optional[str] = None
    
    def with_title(self, title: str) -> 'TaskBuilder':
        """Set task title (required)"""
        self._title = title
        return self  # Return self for chaining!
    
    def with_description(self, description: str) -> 'TaskBuilder':
        """Set task description"""
        self._description = description
        return self
    
    def make_normal(self) -> 'TaskBuilder':
        """Make this a normal task"""
        self._task_type = "normal"
        return self
    
    def make_urgent(self, deadline_hours: int = 24) -> 'TaskBuilder':
        """Make this an urgent task with deadline"""
        self._task_type = "urgent"
        self._deadline = datetime.now() + timedelta(hours=deadline_hours)
        return self
    
    def make_urgent_with_deadline(self, deadline: datetime) -> 'TaskBuilder':
        """Make this an urgent task with specific deadline"""
        self._task_type = "urgent"
        self._deadline = deadline
        return self
    
    def make_recurring(self, frequency_days: int = 7) -> 'TaskBuilder':
        """Make this a recurring task"""
        self._task_type = "recurring"
        self._frequency_days = frequency_days
        return self
    
    def make_important(self) -> 'TaskBuilder':
        """Make this an important task"""
        self._task_type = "important"
        return self
    
    def add_tag(self, tag: str) -> 'TaskBuilder':
        """Add a tag to the task"""
        self._tags.add(tag)
        return self
    
    def add_tags(self, *tags: str) -> 'TaskBuilder':
        """Add multiple tags at once"""
        self._tags.update(tags)
        return self
    
    def add_note(self, note: str) -> 'TaskBuilder':
        """Add a note to the task"""
        self._notes.append(note)
        return self
    
    def boost_priority(self, levels: int = 1) -> 'TaskBuilder':
        """Boost task priority"""
        self._priority_boost += levels
        return self
    
    def add_reminder(self, hours_from_now: int) -> 'TaskBuilder':
        """Add a reminder"""
        self._reminder_time = datetime.now() + timedelta(hours=hours_from_now)
        return self
    
    def add_reminder_at(self, reminder_time: datetime) -> 'TaskBuilder':
        """Add a reminder at specific time"""
        self._reminder_time = reminder_time
        return self
    
    def delegate_to(self, person: str) -> 'TaskBuilder':
        """Delegate task to someone"""
        self._delegated_to = person
        return self
    
    def validate(self) -> bool:
        """Validate the builder state"""
        if not self._title or not self._title.strip():
            raise ValueError("Task title is required")
        
        if self._task_type == "urgent" and self._deadline and self._deadline < datetime.now():
            raise ValueError("Urgent task deadline cannot be in the past")
        
        if self._task_type == "recurring" and self._frequency_days < 1:
            raise ValueError("Recurring task frequency must be at least 1 day")
        
        return True
    
    def build(self) -> 'Task':
        self.validate()
        
        task = TaskFactory().create_task(
            self._title,
            self._description,
            self._task_type,
            self._deadline,
            self._frequency_days
        )
        
        # Apply decorators - all already defined!
        if self._priority_boost > 0:
            task = PriorityBoostDecorator(task, self._priority_boost)
        
        for tag in self._tags:
            task = TagDecorator(task, tag)
        
        if self._notes:
            combined_notes = "\n".join(self._notes)
            task = NotesDecorator(task, combined_notes)
        
        if self._reminder_time:
            task = ReminderDecorator(task, self._reminder_time)
        
        if self._delegated_to:
            task = DelegatedDecorator(task, self._delegated_to)
        
        return task
    
    def reset(self) -> 'TaskBuilder':
        """Reset builder to initial state"""
        self.__init__()
        return self
    
    # ========== PRESET BUILDERS (Convenience Methods) ==========
    
    @staticmethod
    def quick_task(title: str) -> 'TaskBuilder':
        """Quick way to create a simple task"""
        return TaskBuilder().with_title(title)
    
    @staticmethod
    def urgent_bug(title: str, hours: int = 4) -> 'TaskBuilder':
        """Preset for urgent bug fixes"""
        return (TaskBuilder()
                .with_title(title)
                .make_urgent(deadline_hours=hours)
                .add_tags("bug", "urgent")
                .boost_priority(2))
    
    @staticmethod
    def daily_task(title: str) -> 'TaskBuilder':
        """Preset for daily recurring tasks"""
        return (TaskBuilder()
                .with_title(title)
                .make_recurring(frequency_days=1)
                .add_tag("daily"))
    
    @staticmethod
    def weekly_task(title: str) -> 'TaskBuilder':
        """Preset for weekly recurring tasks"""
        return (TaskBuilder()
                .with_title(title)
                .make_recurring(frequency_days=7)
                .add_tag("weekly"))
    
    @staticmethod
    def team_task(title: str, assignee: str) -> 'TaskBuilder':
        """Preset for team tasks"""
        return (TaskBuilder()
                .with_title(title)
                .delegate_to(assignee)
                .add_tag("team"))


# ============================================================================
# SINGLETON PATTERN (From Phase 7)
# ============================================================================

class SingletonMeta(type):
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class ConfigManager(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config: Dict[str, Any] = {
                'urgent_deadline_hours': 24,
                'max_undo_history': 50,
                'enable_notifications': True,
                'date_format': '%Y-%m-%d %H:%M',
                'auto_save': False,
                'theme': 'default'
            }
            self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()


class TaskLogger(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._logs: List[Dict[str, Any]] = []
            self._max_logs = 1000
            self._enabled = True
            self._initialized = True
    
    def log(self, level: str, message: str, **kwargs):
        if not self._enabled:
            return
        
        entry = {
            'timestamp': datetime.now(),
            'level': level.upper(),
            'message': message,
            'data': kwargs
        }
        self._logs.append(entry)
        
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]
    
    def info(self, message: str, **kwargs):
        self.log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log('WARNING', message, **kwargs)
    
    def get_logs(self, level: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        logs = self._logs
        if level:
            logs = [log for log in logs if log['level'] == level.upper()]
        return logs[-limit:]
    
    def clear(self):
        self._logs.clear()


class GlobalStats(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._stats = {
                'tasks_created': 0,
                'tasks_completed': 0,
                'tasks_deleted': 0,
                'commands_executed': 0,
                'undos_performed': 0,
                'tasks_built': 0,  # NEW: Track builder usage
                'app_start_time': datetime.now()
            }
            self._initialized = True
    
    def increment(self, stat_name: str, amount: int = 1):
        if stat_name in self._stats:
            self._stats[stat_name] += amount
    
    def get(self, stat_name: str) -> Any:
        return self._stats.get(stat_name, 0)
    
    def get_all(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        runtime = datetime.now() - stats['app_start_time']
        stats['runtime_seconds'] = int(runtime.total_seconds())
        return stats


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
        
        TaskLogger().info(f"Task created: {self.title}", task_id=self.id)
        GlobalStats().increment('tasks_created')
    
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
            GlobalStats().increment('tasks_completed')
    
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
    def __init__(self, title: str, description: str = "", deadline: Optional[datetime] = None):
        config = ConfigManager()
        default_hours = config.get('urgent_deadline_hours', 24)
        
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
        date_format = config.get('date_format', '%Y-%m-%d %H:%M')
        
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
            GlobalStats().increment('tasks_completed')
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
        GlobalStats().increment('decorations_applied')
    
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
        boosted_value = min(base_priority.value + self.boost_levels, TaskPriority.CRITICAL.value)
        return TaskPriority(boosted_value)
    
    def get_display_icon(self) -> str:
        base_icon = self._wrapped_task.get_display_icon()
        return f"{base_icon}🔺"
    
    def __str__(self) -> str:
        base = str(self._wrapped_task)
        return f"{base} [BOOSTED]"


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
        date_format = config.get('date_format', '%Y-%m-%d %H:%M')
        
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
        return f"{base} (→ {self.delegated_to})"


# ============================================================================
# FACTORY PATTERN (From Phase 5)
# ============================================================================

class TaskFactory:
    URGENT_KEYWORDS = ['urgent', 'asap', 'critical', 'emergency', 'immediately']
    RECURRING_KEYWORDS = ['daily', 'weekly', 'monthly', 'every', 'recurring']
    IMPORTANT_KEYWORDS = ['important', 'priority', 'key', 'crucial', 'vital']
    
    @staticmethod
    def create_task(title: str, description: str = "", task_type: str = "auto", 
                   deadline: Optional[datetime] = None, frequency_days: int = 7) -> Task:
        
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
        
        if not config.get('enable_notifications', True):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if event == "task_added":
            task_type = data.get('type', 'task')
            message = f"🔔 [{timestamp}] New {task_type} added: '{data['title']}'"
        elif event == "task_built":
            message = f"🔔 [{timestamp}] Task built: '{data['title']}'"
        elif event == "task_completed":
            message = f"🔔 [{timestamp}] Task completed: '{data['title']}'"
        elif event == "task_deleted":
            message = f"🔔 [{timestamp}] Task deleted: '{data['title']}'"
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
        return "Priority"


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


# ============================================================================
# COMMAND PATTERN (Simplified for space)
# ============================================================================

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass


class AddTaskCommand(Command):
    def __init__(self, task_manager: 'TaskManager', task: Task):
        self.task_manager = task_manager
        self.task = task
    
    def execute(self):
        self.task_manager.tasks.append(self.task)
        self.task_manager.notify("task_added", {
            'id': self.task.id,
            'title': self.task.title,
            'type': self.task.get_type_name()
        })
        GlobalStats().increment('commands_executed')
    
    def undo(self):
        self.task_manager.tasks.remove(self.task)
        GlobalStats().increment('undos_performed')


class CommandHistory:
    def __init__(self):
        self.history: List[Command] = []
    
    def execute_command(self, command: Command):
        command.execute()
        self.history.append(command)
    
    def undo(self) -> bool:
        if not self.history:
            return False
        command = self.history.pop()
        command.undo()
        return True
    
    def can_undo(self) -> bool:
        return len(self.history) > 0


# ============================================================================
# TASK MANAGER
# ============================================================================

class TaskManager(Subject):
    def __init__(self):
        super().__init__()
        self.tasks: List[Task] = []
        self.command_history = CommandHistory()
        self.sort_strategy: SortStrategy = SortByPriorityStrategy()
        self.filter_strategy: FilterStrategy = ShowAllFilter()
        self.display_strategy: DisplayStrategy = SimpleDisplayStrategy()
        
        TaskLogger().info("TaskManager initialized")
    
    def add_task_from_builder(self, task: Task) -> bool:
        """Add a task created by TaskBuilder"""
        command = AddTaskCommand(self, task)
        self.command_history.execute_command(command)
        GlobalStats().increment('tasks_built')
        self.notify("task_built", {'title': task.title})
        return True
    
    def undo(self) -> bool:
        return self.command_history.undo()
    
    def display_tasks(self):
        tasks = self.filter_strategy.filter(self.tasks)
        tasks = self.sort_strategy.sort(tasks)
        print(f"\n--- Tasks ({self.filter_strategy.name()}, {self.sort_strategy.name()}) ---")
        self.display_strategy.display(tasks)
    
    def get_stats(self) -> dict:
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.completed)
        
        return {
            'total': total,
            'completed': completed,
            'pending': total - completed
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

class TaskManagerCLI:
    def __init__(self):
        self.manager = TaskManager()
        
        notification_observer = NotificationObserver()
        self.manager.attach(notification_observer)
        
        self.running = True
        
        TaskLogger().info("TaskManagerCLI started")
    
    def display_menu(self):
        print("\n" + "="*75)
        print("📋 TASK MANAGER - Phase 8: Builder Pattern")
        print("="*75)
        print("Builder Pattern (NEW!):")
        print("  1. 🏗️  Build Task (Fluent Interface)")
        print("  2. ⚡ Quick Task (Simple)")
        print("  3. 🐛 Urgent Bug Preset")
        print("  4. 📅 Daily Task Preset")
        print("  5. �� Team Task Preset)
        print("\nBasic:")
        print("  6. View Tasks")
        print("  7. Complete Task")
        print("  8. Delete Task (not implemented)")
        print("\nInfo:")
        print("  9. View Statistics")
        print(f"  10. Undo {'✅' if self.manager.command_history.can_undo() else '❌'}")
        print("  11. Exit")
        print("="*75)
    
    def build_task_interactive(self):
        """Interactive builder with fluent interface"""
        print("\n--- 🏗️  Task Builder (Fluent Interface) ---")
        print("Let's build a task step by step!\n")
        
        # Start builder
        builder = TaskBuilder()
        
        # Title (required)
        title = input("Task title: ").strip()
        if not title:
            print("❌ Title is required!")
            return
        builder.with_title(title)
        
        # Description
        desc = input("Description (optional): ").strip()
        if desc:
            builder.with_description(desc)
        
        # Task type
        print("\nTask type:")
        print("  1. Normal")
        print("  2. Urgent (with deadline)")
        print("  3. Recurring")
        print("  4. Important")
        
        type_choice = input("Choose (1-4, default 1): ").strip() or "1"
        
        if type_choice == "2":
            hours = input("Deadline in hours (default 24): ").strip()
            hours = int(hours) if hours.isdigit() else 24
            builder.make_urgent(deadline_hours=hours)
        elif type_choice == "3":
            days = input("Repeat every X days (default 7): ").strip()
            days = int(days) if days.isdigit() else 7
            builder.make_recurring(frequency_days=days)
        elif type_choice == "4":
            builder.make_important()
        else:
            builder.make_normal()
        
        # Tags
        print("\nAdd tags? (comma-separated, or press Enter to skip)")
        tags_input = input("Tags: ").strip()
        if tags_input:
            tags = [t.strip() for t in tags_input.split(',')]
            builder.add_tags(*tags)
        
        # Priority boost
        boost = input("\nBoost priority? (0-3, default 0): ").strip()
        if boost and boost.isdigit() and int(boost) > 0:
            builder.boost_priority(int(boost))
        
        # Reminder
        reminder = input("\nAdd reminder in hours? (Enter to skip): ").strip()
        if reminder and reminder.isdigit():
            builder.add_reminder(int(reminder))
        
        # Delegation
        delegate = input("\nDelegate to someone? (Enter to skip): ").strip()
        if delegate:
            builder.delegate_to(delegate)
        
        # Notes
        note = input("\nAdd a note? (Enter to skip): ").strip()
        if note:
            builder.add_note(note)
        
        # Build!
        print("\n🔨 Building task...")
        try:
            task = builder.build()
            self.manager.add_task_from_builder(task)
            print(f"\n✅ Task built successfully!")
            print(f"   {task}")
        except ValueError as e:
            print(f"\n❌ Build failed: {e}")
    
    def quick_task_interactive(self):
        """Quick task creation"""
        print("\n--- ⚡ Quick Task ---")
        title = input("Task title: ").strip()
        
        if not title:
            print("❌ Title is required!")
            return
        
        try:
            task = TaskBuilder.quick_task(title).build()
            self.manager.add_task_from_builder(task)
            print(f"✅ Quick task created!")
        except ValueError as e:
            print(f"❌ Failed: {e}")
    
    def urgent_bug_preset_interactive(self):
        """Create urgent bug using preset"""
        print("\n--- 🐛 Urgent Bug Preset ---")
        title = input("Bug description: ").strip()
        
        if not title:
            print("❌ Title is required!")
            return
        
        hours = input("Critical? How many hours? (default 4): ").strip()
        hours = int(hours) if hours.isdigit() else 4
        
        try:
            task = (TaskBuilder.urgent_bug(title, hours)
                   .with_description("Production bug - needs immediate attention")
                   .build())
            
            self.manager.add_task_from_builder(task)
            print(f"✅ Urgent bug task created with {hours}h deadline!")
            print(f"   {task}")
        except ValueError as e:
            print(f"❌ Failed: {e}")
    
    def daily_task_preset_interactive(self):
        """Create daily recurring task"""
        print("\n--- 📅 Daily Task Preset ---")
        title = input("Daily task (e.g., 'Check emails'): ").strip()
        
        if not title:
            print("❌ Title is required!")
            return
        
        try:
            task = TaskBuilder.daily_task(title).build()
            self.manager.add_task_from_builder(task)
            print(f"✅ Daily recurring task created!")
            print(f"   {task}")
        except ValueError as e:
            print(f"❌ Failed: {e}")
    
    def team_task_preset_interactive(self):
        """Create team task"""
        print("\n--- 👥 Team Task Preset ---")
        title = input("Task title: ").strip()
        
        if not title:
            print("❌ Title is required!")
            return
        
        assignee = input("Assign to: ").strip()
        
        if not assignee:
            print("❌ Assignee is required!")
            return
        
        try:
            task = (TaskBuilder.team_task(title, assignee)
                   .add_tag("collaboration")
                   .build())
            
            self.manager.add_task_from_builder(task)
            print(f"✅ Team task created and assigned to {assignee}!")
            print(f"   {task}")
        except ValueError as e:
            print(f"❌ Failed: {e}")
    
    def complete_task_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return
        
        self.manager.display_tasks()
        
        # Simple index-based selection
        try:
            index = int(input("\nEnter task number to complete: ").strip()) - 1
            if 0 <= index < len(self.manager.tasks):
                task = self.manager.tasks[index]
                task.toggle_completion()
                status = "completed" if task.completed else "uncompleted"
                print(f"✅ Task {status}!")
            else:
                print("❌ Invalid task number!")
        except ValueError:
            print("❌ Invalid input!")
    
    def view_statistics(self):
        stats = self.manager.get_stats()
        global_stats = GlobalStats().get_all()
        
        print("\n--- 📊 Statistics ---")
        print(f"Tasks: {stats['total']} total, {stats['completed']} completed, {stats['pending']} pending")
        print(f"\nBuilder Usage:")
        print(f"  Tasks built with Builder: {global_stats['tasks_built']}")
        print(f"  Total tasks created: {global_stats['tasks_created']}")
        print(f"  Commands executed: {global_stats['commands_executed']}")
    
    def run(self):
        print("Welcome to Task Manager - Phase 8: Builder Pattern!")
        print("🏗️  Now featuring fluent task building!")
        print()
        
        # Demonstrate Builder pattern
        print("🔍 Builder Pattern Demo:")
        demo_task = (TaskBuilder()
                    .with_title("Example Task")
                    .make_urgent(deadline_hours=2)
                    .add_tags("demo", "example")
                    .boost_priority(1)
                    .build())
        
        print(f"   Built: {demo_task}")
        print(f"   With method chaining! ✨\n")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-11): ").strip()
            
            if choice == '1':
                self.build_task_interactive()
            elif choice == '2':
                self.quick_task_interactive()
            elif choice == '3':
                self.urgent_bug_preset_interactive()
            elif choice == '4':
                self.daily_task_preset_interactive()
            elif choice == '5':
                self.team_task_preset_interactive()
            elif choice == '6':
                self.manager.display_tasks()
            elif choice == '7':
                self.complete_task_interactive()
            elif choice == '9':
                self.view_statistics()
            elif choice == '10':
                if self.manager.undo():
                    print("✅ Last command undone!")
                else:
                    print("❌ Nothing to undo!")
            elif choice == '11':
                TaskLogger().info("Application shutting down")
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                print("🎓 You've learned 7 design patterns! Keep building! 🚀")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-11.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()
