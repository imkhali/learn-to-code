"""
Task Manager - Phase 2: Observer Pattern
Learn how to implement the Observer pattern for event-driven programming
"""

from datetime import datetime
from typing import List, Optional
from abc import ABC, abstractmethod


# ============================================================================
# OBSERVER PATTERN IMPLEMENTATION
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
        """Display notification based on event type"""
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
        """Record event in history"""
        entry = {
            'timestamp': datetime.now(),
            'event': event,
            'data': data.copy()
        }
        self.history.append(entry)
    
    def get_history(self) -> List[dict]:
        """Return all recorded history"""
        return self.history
    
    def display_history(self, limit: int = 10):
        """Display recent history"""
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


class StatisticsObserver(Observer):
    """Observes events and tracks statistics"""
    
    def __init__(self):
        self.events_count = {
            'task_added': 0,
            'task_completed': 0,
            'task_uncompleted': 0,
            'task_deleted': 0
        }
    
    def update(self, event: str, data: dict):
        """Update statistics based on event"""
        if event in self.events_count:
            self.events_count[event] += 1
    
    def display_stats(self):
        """Display event statistics"""
        print("\n--- 📊 Event Statistics ---")
        print(f"Tasks Added: {self.events_count['task_added']}")
        print(f"Tasks Completed: {self.events_count['task_completed']}")
        print(f"Tasks Reopened: {self.events_count['task_uncompleted']}")
        print(f"Tasks Deleted: {self.events_count['task_deleted']}")


class Subject:
    """Subject that observers can subscribe to"""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        """Subscribe an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        """Unsubscribe an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: str, data: dict):
        """Notify all observers of an event"""
        for observer in self._observers:
            observer.update(event, data)


# ============================================================================
# TASK AND TASK MANAGER (Updated with Observer Pattern)
# ============================================================================

class Task:
    """Represents a single task with all its properties"""
    
    def __init__(self, title: str, description: str = ""):
        self.id = self._generate_id()
        self.title = title
        self.description = description
        self.completed = False
        self.created_at = datetime.now()
    
    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID based on timestamp"""
        return str(int(datetime.now().timestamp() * 1000))
    
    def toggle_completion(self):
        """Toggle the completion status of the task"""
        self.completed = not self.completed
    
    def __str__(self) -> str:
        """String representation of the task"""
        status = "✓" if self.completed else "○"
        title_display = f"[DONE] {self.title}" if self.completed else self.title
        return f"{status} [{self.id}] {title_display}"


class TaskManager(Subject):
    """Manages the collection of tasks with Observer pattern support"""
    
    def __init__(self):
        super().__init__()  # Initialize Subject
        self.tasks: List[Task] = []
    
    # CREATE
    def add_task(self, title: str, description: str = "") -> Task:
        """Add a new task and notify observers"""
        if not title.strip():
            raise ValueError("Task title cannot be empty")
        
        task = Task(title.strip(), description.strip())
        self.tasks.append(task)
        
        # Notify observers about the new task
        self.notify("task_added", {
            'id': task.id,
            'title': task.title,
            'description': task.description
        })
        
        return task
    
    # READ
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks"""
        return self.tasks
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find a task by its ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_pending_tasks(self) -> List[Task]:
        """Get only pending tasks"""
        return [task for task in self.tasks if not task.completed]
    
    def get_completed_tasks(self) -> List[Task]:
        """Get only completed tasks"""
        return [task for task in self.tasks if task.completed]
    
    # UPDATE
    def toggle_task(self, task_id: str) -> bool:
        """Toggle completion status and notify observers"""
        task = self.get_task_by_id(task_id)
        if task:
            was_completed = task.completed
            task.toggle_completion()
            
            # Notify observers
            event = "task_uncompleted" if was_completed else "task_completed"
            self.notify(event, {
                'id': task.id,
                'title': task.title
            })
            
            return True
        return False
    
    # DELETE
    def delete_task(self, task_id: str) -> bool:
        """Delete a task and notify observers"""
        task = self.get_task_by_id(task_id)
        if task:
            # Notify before deletion
            self.notify("task_deleted", {
                'id': task.id,
                'title': task.title
            })
            
            self.tasks.remove(task)
            return True
        return False
    
    # STATS
    def get_stats(self) -> dict:
        """Get statistics about tasks"""
        total = len(self.tasks)
        completed = len(self.get_completed_tasks())
        pending = len(self.get_pending_tasks())
        
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
    """Command-line interface for the Task Manager"""
    
    def __init__(self):
        self.manager = TaskManager()
        
        # Create and attach observers
        self.notification_observer = NotificationObserver()
        self.history_observer = HistoryObserver()
        self.stats_observer = StatisticsObserver()
        
        self.manager.attach(self.notification_observer)
        self.manager.attach(self.history_observer)
        self.manager.attach(self.stats_observer)
        
        self.running = True
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("📋 TASK MANAGER - Phase 2: Observer Pattern")
        print("="*50)
        print("1. Add Task")
        print("2. View All Tasks")
        print("3. View Pending Tasks")
        print("4. View Completed Tasks")
        print("5. Complete/Uncomplete Task")
        print("6. Delete Task")
        print("7. View Statistics")
        print("8. View History")
        print("9. View Event Statistics")
        print("10. Exit")
        print("="*50)
    
    def add_task_interactive(self):
        """Interactive task creation"""
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
    
    def view_tasks(self, task_list: List[Task], title: str):
        """Display a list of tasks"""
        print(f"\n--- {title} ---")
        
        if not task_list:
            print("No tasks to display.")
            return
        
        for i, task in enumerate(task_list, 1):
            print(f"{i}. {task}")
    
    def toggle_task_interactive(self):
        """Interactive task completion toggle"""
        tasks = self.manager.get_all_tasks()
        
        if not tasks:
            print("\n❌ No tasks available!")
            return
        
        self.view_tasks(tasks, "All Tasks")
        
        task_id = input("\nEnter task ID to toggle: ").strip()
        
        if self.manager.toggle_task(task_id):
            task = self.manager.get_task_by_id(task_id)
            status = "completed" if task.completed else "pending"
            print(f"✅ Task marked as {status}!")
        else:
            print("❌ Task not found!")
    
    def delete_task_interactive(self):
        """Interactive task deletion"""
        tasks = self.manager.get_all_tasks()
        
        if not tasks:
            print("\n❌ No tasks available!")
            return
        
        self.view_tasks(tasks, "All Tasks")
        
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
        """Display task statistics"""
        stats = self.manager.get_stats()
        
        print("\n--- 📊 Task Statistics ---")
        print(f"Total Tasks: {stats['total']}")
        print(f"Completed: {stats['completed']}")
        print(f"Pending: {stats['pending']}")
        print(f"Completion Rate: {stats['completion_rate']:.1f}%")
    
    def run(self):
        """Main application loop"""
        print("Welcome to Task Manager - Phase 2!")
        print("🎯 Now with Observer Pattern for real-time notifications!")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-10): ").strip()
            
            if choice == '1':
                self.add_task_interactive()
            elif choice == '2':
                self.view_tasks(self.manager.get_all_tasks(), "All Tasks")
            elif choice == '3':
                self.view_tasks(self.manager.get_pending_tasks(), "Pending Tasks")
            elif choice == '4':
                self.view_tasks(self.manager.get_completed_tasks(), "Completed Tasks")
            elif choice == '5':
                self.toggle_task_interactive()
            elif choice == '6':
                self.delete_task_interactive()
            elif choice == '7':
                self.view_statistics()
            elif choice == '8':
                self.history_observer.display_history()
            elif choice == '9':
                self.stats_observer.display_stats()
            elif choice == '10':
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-10.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()