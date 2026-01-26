"""
Task Manager - Phase 3: Strategy Pattern
Learn how to implement the Strategy pattern for interchangeable algorithms
"""

from datetime import datetime
from typing import List, Optional
from abc import ABC, abstractmethod


# ============================================================================
# OBSERVER PATTERN (From Phase 2)
# ============================================================================

class Observer(ABC):
    """Abstract base class for all observers"""
    
    @abstractmethod
    def update(self, event: str, data: dict):
        """Called when an event occurs"""
        pass


class NotificationObserver(Observer):
    """Observes events and displays notifications"""
    
    def update(self, event: str, data: dict):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if event == "task_added":
            print(f"\n🔔 [{timestamp}] New task added: '{data['title']}'")
        elif event == "task_completed":
            print(f"\n🔔 [{timestamp}] Task completed: '{data['title']}'")
        elif event == "task_uncompleted":
            print(f"\n🔔 [{timestamp}] Task reopened: '{data['title']}'")
        elif event == "task_deleted":
            print(f"\n🔔 [{timestamp}] Task deleted: '{data['title']}'")


class HistoryObserver(Observer):
    """Observes events and records history"""
    
    def __init__(self):
        self.history: List[dict] = []
    
    def update(self, event: str, data: dict):
        entry = {
            'timestamp': datetime.now(),
            'event': event,
            'data': data.copy()
        }
        self.history.append(entry)
    
    def display_history(self, limit: int = 10):
        print(f"\n--- 📜 Recent History (Last {limit}) ---")
        if not self.history:
            print("No history yet.")
            return
        
        recent = self.history[-limit:]
        for i, entry in enumerate(reversed(recent), 1):
            timestamp = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            event = entry['event'].replace('_', ' ').title()
            title = entry['data'].get('title', 'Unknown')
            print(f"{i}. [{timestamp}] {event}: '{title}'")


class Subject:
    """Subject that observers can subscribe to"""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: str, data: dict):
        for observer in self._observers:
            observer.update(event, data)


# ============================================================================
# STRATEGY PATTERN - SORTING STRATEGIES
# ============================================================================

class SortStrategy(ABC):
    """Abstract base class for sorting strategies"""
    
    @abstractmethod
    def sort(self, tasks: List['Task']) -> List['Task']:
        """Sort tasks according to strategy"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy"""
        pass


class SortByDateStrategy(SortStrategy):
    """Sort tasks by creation date (newest first)"""
    
    def sort(self, tasks: List['Task']) -> List['Task']:
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def name(self) -> str:
        return "Date (Newest First)"


class SortByDateOldestStrategy(SortStrategy):
    """Sort tasks by creation date (oldest first)"""
    
    def sort(self, tasks: List['Task']) -> List['Task']:
        return sorted(tasks, key=lambda t: t.created_at)
    
    def name(self) -> str:
        return "Date (Oldest First)"


class SortByTitleStrategy(SortStrategy):
    """Sort tasks alphabetically by title"""
    
    def sort(self, tasks: List['Task']) -> List['Task']:
        return sorted(tasks, key=lambda t: t.title.lower())
    
    def name(self) -> str:
        return "Title (A-Z)"


class SortByCompletionStrategy(SortStrategy):
    """Sort tasks by completion status (pending first)"""
    
    def sort(self, tasks: List['Task']) -> List['Task']:
        return sorted(tasks, key=lambda t: t.completed)
    
    def name(self) -> str:
        return "Status (Pending First)"


# ============================================================================
# STRATEGY PATTERN - FILTERING STRATEGIES
# ============================================================================

class FilterStrategy(ABC):
    """Abstract base class for filtering strategies"""
    
    @abstractmethod
    def filter(self, tasks: List['Task']) -> List['Task']:
        """Filter tasks according to strategy"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy"""
        pass


class ShowAllFilter(FilterStrategy):
    """Show all tasks"""
    
    def filter(self, tasks: List['Task']) -> List['Task']:
        return tasks
    
    def name(self) -> str:
        return "All Tasks"


class ShowPendingFilter(FilterStrategy):
    """Show only pending tasks"""
    
    def filter(self, tasks: List['Task']) -> List['Task']:
        return [t for t in tasks if not t.completed]
    
    def name(self) -> str:
        return "Pending Only"


class ShowCompletedFilter(FilterStrategy):
    """Show only completed tasks"""
    
    def filter(self, tasks: List['Task']) -> List['Task']:
        return [t for t in tasks if t.completed]
    
    def name(self) -> str:
        return "Completed Only"


class ShowRecentFilter(FilterStrategy):
    """Show tasks created in the last 24 hours"""
    
    def filter(self, tasks: List['Task']) -> List['Task']:
        now = datetime.now()
        return [t for t in tasks if (now - t.created_at).days < 1]
    
    def name(self) -> str:
        return "Recent (Last 24h)"


# ============================================================================
# STRATEGY PATTERN - DISPLAY STRATEGIES
# ============================================================================

class DisplayStrategy(ABC):
    """Abstract base class for display strategies"""
    
    @abstractmethod
    def display(self, tasks: List['Task']):
        """Display tasks according to strategy"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy"""
        pass


class SimpleDisplayStrategy(DisplayStrategy):
    """Simple one-line display"""
    
    def display(self, tasks: List['Task']):
        if not tasks:
            print("No tasks to display.")
            return
        
        for i, task in enumerate(tasks, 1):
            status = "✓" if task.completed else "○"
            print(f"{i}. {status} {task.title}")
    
    def name(self) -> str:
        return "Simple"


class DetailedDisplayStrategy(DisplayStrategy):
    """Detailed multi-line display"""
    
    def display(self, tasks: List['Task']):
        if not tasks:
            print("No tasks to display.")
            return
        
        for i, task in enumerate(tasks, 1):
            status = "✓ DONE" if task.completed else "○ PENDING"
            print(f"\n{i}. [{task.id}] {status}")
            print(f"   Title: {task.title}")
            if task.description:
                print(f"   Description: {task.description}")
            print(f"   Created: {task.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    def name(self) -> str:
        return "Detailed"


class CompactDisplayStrategy(DisplayStrategy):
    """Very compact display with emojis"""
    
    def display(self, tasks: List['Task']):
        if not tasks:
            print("No tasks to display.")
            return
        
        for i, task in enumerate(tasks, 1):
            emoji = "✅" if task.completed else "⏳"
            title = task.title[:40] + "..." if len(task.title) > 40 else task.title
            print(f"{i}. {emoji} {title}")
    
    def name(self) -> str:
        return "Compact"


# ============================================================================
# TASK AND TASK MANAGER
# ============================================================================

class Task:
    """Represents a single task"""
    
    def __init__(self, title: str, description: str = ""):
        self.id = self._generate_id()
        self.title = title
        self.description = description
        self.completed = False
        self.created_at = datetime.now()
    
    @staticmethod
    def _generate_id() -> str:
        return str(int(datetime.now().timestamp() * 1000))
    
    def toggle_completion(self):
        self.completed = not self.completed


class TaskManager(Subject):
    """Manages tasks with Strategy pattern support"""
    
    def __init__(self):
        super().__init__()
        self.tasks: List[Task] = []
        
        # Default strategies
        self.sort_strategy: SortStrategy = SortByDateStrategy()
        self.filter_strategy: FilterStrategy = ShowAllFilter()
        self.display_strategy: DisplayStrategy = SimpleDisplayStrategy()
    
    # Strategy setters
    def set_sort_strategy(self, strategy: SortStrategy):
        """Change the sorting strategy"""
        self.sort_strategy = strategy
    
    def set_filter_strategy(self, strategy: FilterStrategy):
        """Change the filtering strategy"""
        self.filter_strategy = strategy
    
    def set_display_strategy(self, strategy: DisplayStrategy):
        """Change the display strategy"""
        self.display_strategy = strategy
    
    # Get tasks with current strategies applied
    def get_tasks_with_strategies(self) -> List[Task]:
        """Apply current filter and sort strategies"""
        filtered = self.filter_strategy.filter(self.tasks)
        sorted_tasks = self.sort_strategy.sort(filtered)
        return sorted_tasks
    
    def display_tasks(self):
        """Display tasks using current display strategy"""
        tasks = self.get_tasks_with_strategies()
        print(f"\n--- Tasks ({self.filter_strategy.name()}, {self.sort_strategy.name()}) ---")
        self.display_strategy.display(tasks)
    
    # CRUD operations
    def add_task(self, title: str, description: str = "") -> Task:
        if not title.strip():
            raise ValueError("Task title cannot be empty")
        
        task = Task(title.strip(), description.strip())
        self.tasks.append(task)
        
        self.notify("task_added", {
            'id': task.id,
            'title': task.title,
            'description': task.description
        })
        
        return task
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def toggle_task(self, task_id: str) -> bool:
        task = self.get_task_by_id(task_id)
        if task:
            was_completed = task.completed
            task.toggle_completion()
            
            event = "task_uncompleted" if was_completed else "task_completed"
            self.notify(event, {'id': task.id, 'title': task.title})
            
            return True
        return False
    
    def delete_task(self, task_id: str) -> bool:
        task = self.get_task_by_id(task_id)
        if task:
            self.notify("task_deleted", {'id': task.id, 'title': task.title})
            self.tasks.remove(task)
            return True
        return False
    
    def get_stats(self) -> dict:
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.completed)
        pending = total - completed
        
        return {
            'total': total,
            'completed': completed,
            'pending': pending,
            'completion_rate': (completed / total * 100) if total > 0 else 0
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

class TaskManagerCLI:
    """Command-line interface with Strategy pattern"""
    
    def __init__(self):
        self.manager = TaskManager()
        
        # Observers
        self.notification_observer = NotificationObserver()
        self.history_observer = HistoryObserver()
        
        self.manager.attach(self.notification_observer)
        self.manager.attach(self.history_observer)
        
        # Available strategies
        self.sort_strategies = {
            '1': SortByDateStrategy(),
            '2': SortByDateOldestStrategy(),
            '3': SortByTitleStrategy(),
            '4': SortByCompletionStrategy()
        }
        
        self.filter_strategies = {
            '1': ShowAllFilter(),
            '2': ShowPendingFilter(),
            '3': ShowCompletedFilter(),
            '4': ShowRecentFilter()
        }
        
        self.display_strategies = {
            '1': SimpleDisplayStrategy(),
            '2': DetailedDisplayStrategy(),
            '3': CompactDisplayStrategy()
        }
        
        self.running = True
    
    def display_menu(self):
        print("\n" + "="*60)
        print("📋 TASK MANAGER - Phase 3: Strategy Pattern")
        print("="*60)
        print("Tasks:")
        print("  1. Add Task")
        print("  2. View Tasks (with current strategies)")
        print("  3. Complete/Uncomplete Task")
        print("  4. Delete Task")
        print("\nStrategies:")
        print("  5. Change Sort Strategy")
        print("  6. Change Filter Strategy")
        print("  7. Change Display Strategy")
        print("\nInfo:")
        print("  8. View Statistics")
        print("  9. View History")
        print("  10. View Current Strategies")
        print("  11. Exit")
        print("="*60)
    
    def change_sort_strategy(self):
        print("\n--- Sort Strategies ---")
        print("1. Date (Newest First)")
        print("2. Date (Oldest First)")
        print("3. Title (A-Z)")
        print("4. Status (Pending First)")
        
        choice = input("\nChoose sort strategy (1-4): ").strip()
        
        if choice in self.sort_strategies:
            self.manager.set_sort_strategy(self.sort_strategies[choice])
            print(f"✅ Sort strategy changed to: {self.sort_strategies[choice].name()}")
        else:
            print("❌ Invalid choice!")
    
    def change_filter_strategy(self):
        print("\n--- Filter Strategies ---")
        print("1. All Tasks")
        print("2. Pending Only")
        print("3. Completed Only")
        print("4. Recent (Last 24h)")
        
        choice = input("\nChoose filter strategy (1-4): ").strip()
        
        if choice in self.filter_strategies:
            self.manager.set_filter_strategy(self.filter_strategies[choice])
            print(f"✅ Filter strategy changed to: {self.filter_strategies[choice].name()}")
        else:
            print("❌ Invalid choice!")
    
    def change_display_strategy(self):
        print("\n--- Display Strategies ---")
        print("1. Simple")
        print("2. Detailed")
        print("3. Compact")
        
        choice = input("\nChoose display strategy (1-3): ").strip()
        
        if choice in self.display_strategies:
            self.manager.set_display_strategy(self.display_strategies[choice])
            print(f"✅ Display strategy changed to: {self.display_strategies[choice].name()}")
        else:
            print("❌ Invalid choice!")
    
    def view_current_strategies(self):
        print("\n--- 🎯 Current Strategies ---")
        print(f"Sort: {self.manager.sort_strategy.name()}")
        print(f"Filter: {self.manager.filter_strategy.name()}")
        print(f"Display: {self.manager.display_strategy.name()}")
    
    def add_task_interactive(self):
        print("\n--- Add New Task ---")
        title = input("Enter task title: ").strip()
        
        if not title:
            print("❌ Task title cannot be empty!")
            return
        
        description = input("Enter description (optional): ").strip()
        
        try:
            task = self.manager.add_task(title, description)
            print(f"✅ Task added successfully! ID: {task.id}")
        except ValueError as e:
            print(f"❌ Error: {e}")
    
    def toggle_task_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return
        
        self.manager.display_tasks()
        task_id = input("\nEnter task ID to toggle: ").strip()
        
        if self.manager.toggle_task(task_id):
            task = self.manager.get_task_by_id(task_id)
            status = "completed" if task.completed else "pending"
            print(f"✅ Task marked as {status}!")
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
                print("✅ Task deleted successfully!")
            else:
                print("❌ Task not found!")
        else:
            print("Deletion cancelled.")
    
    def view_statistics(self):
        stats = self.manager.get_stats()
        print("\n--- 📊 Task Statistics ---")
        print(f"Total Tasks: {stats['total']}")
        print(f"Completed: {stats['completed']}")
        print(f"Pending: {stats['pending']}")
        print(f"Completion Rate: {stats['completion_rate']:.1f}%")
    
    def run(self):
        print("Welcome to Task Manager - Phase 3!")
        print("🎯 Now with Strategy Pattern for flexible sorting, filtering, and display!")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-11): ").strip()
            
            if choice == '1':
                self.add_task_interactive()
            elif choice == '2':
                self.manager.display_tasks()
            elif choice == '3':
                self.toggle_task_interactive()
            elif choice == '4':
                self.delete_task_interactive()
            elif choice == '5':
                self.change_sort_strategy()
            elif choice == '6':
                self.change_filter_strategy()
            elif choice == '7':
                self.change_display_strategy()
            elif choice == '8':
                self.view_statistics()
            elif choice == '9':
                self.history_observer.display_history()
            elif choice == '10':
                self.view_current_strategies()
            elif choice == '11':
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-11.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()