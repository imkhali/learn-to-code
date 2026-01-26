"""
Task Manager - Phase 4: Command Pattern
Learn how to implement the Command pattern for undo/redo functionality
"""

from datetime import datetime
from typing import List, Optional
from abc import ABC, abstractmethod
from copy import deepcopy


# ============================================================================
# OBSERVER PATTERN (From Phase 2)
# ============================================================================

class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: dict):
        pass


class NotificationObserver(Observer):
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
        elif event == "command_executed":
            print(f"\n🔔 [{timestamp}] Command executed: {data['command']}")
        elif event == "command_undone":
            print(f"\n🔔 [{timestamp}] Command undone: {data['command']}")


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
# COMMAND PATTERN IMPLEMENTATION
# ============================================================================

class Command(ABC):
    """Abstract base class for all commands"""
    
    @abstractmethod
    def execute(self):
        """Execute the command"""
        pass
    
    @abstractmethod
    def undo(self):
        """Undo the command"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of the command"""
        pass


class AddTaskCommand(Command):
    """Command to add a task"""
    
    def __init__(self, task_manager, title: str, description: str = ""):
        self.task_manager = task_manager
        self.title = title
        self.description = description
        self.task = None  # Will store the created task
    
    def execute(self):
        """Add the task"""
        self.task = self.task_manager._add_task_internal(self.title, self.description)
        self.task_manager.notify("command_executed", {
            'command': self.get_description()
        })
    
    def undo(self):
        """Remove the task that was added"""
        if self.task:
            self.task_manager._delete_task_internal(self.task.id)
            self.task_manager.notify("command_undone", {
                'command': self.get_description()
            })
    
    def get_description(self) -> str:
        return f"Add Task: '{self.title}'"


class DeleteTaskCommand(Command):
    """Command to delete a task"""
    
    def __init__(self, task_manager, task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id
        self.deleted_task = None  # Will store task data for undo
    
    def execute(self):
        """Delete the task (but save it for undo)"""
        task = self.task_manager.get_task_by_id(self.task_id)
        if task:
            # Save a deep copy for undo
            self.deleted_task = deepcopy(task)
            self.task_manager._delete_task_internal(self.task_id)
            self.task_manager.notify("command_executed", {
                'command': self.get_description()
            })
            return True
        return False
    
    def undo(self):
        """Restore the deleted task"""
        if self.deleted_task:
            # Re-add the task with original data
            self.task_manager.tasks.append(self.deleted_task)
            self.task_manager.notify("command_undone", {
                'command': self.get_description()
            })
    
    def get_description(self) -> str:
        title = self.deleted_task.title if self.deleted_task else "Unknown"
        return f"Delete Task: '{title}'"


class ToggleTaskCommand(Command):
    """Command to toggle task completion"""
    
    def __init__(self, task_manager, task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id
        self.task = None
    
    def execute(self):
        """Toggle the task completion status"""
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
            return True
        return False
    
    def undo(self):
        """Toggle back to original state"""
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


class MacroCommand(Command):
    """Command that executes multiple commands"""
    
    def __init__(self, commands: List[Command], description: str = "Macro"):
        self.commands = commands
        self.description = description
    
    def execute(self):
        """Execute all commands in order"""
        for command in self.commands:
            command.execute()
    
    def undo(self):
        """Undo all commands in reverse order"""
        for command in reversed(self.commands):
            command.undo()
    
    def get_description(self) -> str:
        return f"Macro: {self.description} ({len(self.commands)} commands)"


class CommandHistory:
    """Manages command history for undo/redo"""
    
    def __init__(self):
        self.history: List[Command] = []  # Executed commands
        self.undo_stack: List[Command] = []  # Undone commands for redo
    
    def execute_command(self, command: Command):
        """Execute a command and add to history"""
        command.execute()
        self.history.append(command)
        # Clear redo stack when new command is executed
        self.undo_stack.clear()
    
    def undo(self) -> bool:
        """Undo the last command"""
        if not self.history:
            return False
        
        command = self.history.pop()
        command.undo()
        self.undo_stack.append(command)
        return True
    
    def redo(self) -> bool:
        """Redo the last undone command"""
        if not self.undo_stack:
            return False
        
        command = self.undo_stack.pop()
        command.execute()
        self.history.append(command)
        return True
    
    def get_history(self, limit: int = 10) -> List[Command]:
        """Get recent command history"""
        return self.history[-limit:]
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.history) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.undo_stack) > 0
    
    def clear(self):
        """Clear all history"""
        self.history.clear()
        self.undo_stack.clear()


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
        return str(int(datetime.now().timestamp() * 1000000))  # More unique
    
    def toggle_completion(self):
        self.completed = not self.completed
    
    def __str__(self) -> str:
        status = "✓" if self.completed else "○"
        title_display = f"[DONE] {self.title}" if self.completed else self.title
        return f"{status} [{self.id}] {title_display}"


class TaskManager(Subject):
    """Manages tasks with Command pattern support"""
    
    def __init__(self):
        super().__init__()
        self.tasks: List[Task] = []
        self.command_history = CommandHistory()
    
    # ========== COMMAND-BASED OPERATIONS ==========
    
    def add_task(self, title: str, description: str = "") -> bool:
        """Add task using command pattern"""
        if not title.strip():
            return False
        
        command = AddTaskCommand(self, title, description)
        self.command_history.execute_command(command)
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """Delete task using command pattern"""
        command = DeleteTaskCommand(self, task_id)
        result = command.execute()
        if result:
            self.command_history.history.append(command)
            self.command_history.undo_stack.clear()
        return result
    
    def toggle_task(self, task_id: str) -> bool:
        """Toggle task using command pattern"""
        command = ToggleTaskCommand(self, task_id)
        result = command.execute()
        if result:
            self.command_history.history.append(command)
            self.command_history.undo_stack.clear()
        return result
    
    def execute_macro(self, task_ids: List[str], description: str = "Batch Operation"):
        """Execute multiple toggle commands at once"""
        commands = [ToggleTaskCommand(self, tid) for tid in task_ids]
        macro = MacroCommand(commands, description)
        self.command_history.execute_command(macro)
    
    # ========== UNDO/REDO ==========
    
    def undo(self) -> bool:
        """Undo last command"""
        return self.command_history.undo()
    
    def redo(self) -> bool:
        """Redo last undone command"""
        return self.command_history.redo()
    
    def can_undo(self) -> bool:
        return self.command_history.can_undo()
    
    def can_redo(self) -> bool:
        return self.command_history.can_redo()
    
    # ========== INTERNAL OPERATIONS (Used by Commands) ==========
    
    def _add_task_internal(self, title: str, description: str) -> Task:
        """Internal method to add task (used by AddTaskCommand)"""
        task = Task(title.strip(), description.strip())
        self.tasks.append(task)
        self.notify("task_added", {
            'id': task.id,
            'title': task.title,
            'description': task.description
        })
        return task
    
    def _delete_task_internal(self, task_id: str) -> bool:
        """Internal method to delete task (used by DeleteTaskCommand)"""
        task = self.get_task_by_id(task_id)
        if task:
            self.notify("task_deleted", {'id': task.id, 'title': task.title})
            self.tasks.remove(task)
            return True
        return False
    
    # ========== QUERY OPERATIONS ==========
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_all_tasks(self) -> List[Task]:
        return self.tasks
    
    def get_pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks if not t.completed]
    
    def get_completed_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.completed]
    
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
    """Command-line interface with Command pattern"""
    
    def __init__(self):
        self.manager = TaskManager()
        
        # Observers
        self.notification_observer = NotificationObserver()
        self.manager.attach(self.notification_observer)
        
        self.running = True
    
    def display_menu(self):
        print("\n" + "="*60)
        print("📋 TASK MANAGER - Phase 4: Command Pattern")
        print("="*60)
        print("Tasks:")
        print("  1. Add Task")
        print("  2. View All Tasks")
        print("  3. Complete/Uncomplete Task")
        print("  4. Delete Task")
        print("  5. Batch Complete/Uncomplete (Macro)")
        print("\nCommand History:")
        print(f"  6. Undo {'✅' if self.manager.can_undo() else '❌'}")
        print(f"  7. Redo {'✅' if self.manager.can_redo() else '❌'}")
        print("  8. View Command History")
        print("\nInfo:")
        print("  9. View Statistics")
        print("  10. Exit")
        print("="*60)
    
    def add_task_interactive(self):
        print("\n--- Add New Task ---")
        title = input("Enter task title: ").strip()
        
        if not title:
            print("❌ Task title cannot be empty!")
            return
        
        description = input("Enter description (optional): ").strip()
        
        if self.manager.add_task(title, description):
            print(f"✅ Task added successfully!")
        else:
            print("❌ Failed to add task!")
    
    def view_tasks(self):
        tasks = self.manager.get_all_tasks()
        print("\n--- All Tasks ---")
        
        if not tasks:
            print("No tasks to display.")
            return
        
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
    
    def toggle_task_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return
        
        self.view_tasks()
        task_id = input("\nEnter task ID to toggle: ").strip()
        
        if self.manager.toggle_task(task_id):
            print("✅ Task toggled!")
        else:
            print("❌ Task not found!")
    
    def delete_task_interactive(self):
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return
        
        self.view_tasks()
        task_id = input("\nEnter task ID to delete: ").strip()
        confirm = input("Are you sure? (y/n): ").strip().lower()
        
        if confirm == 'y':
            if self.manager.delete_task(task_id):
                print("✅ Task deleted successfully! (Can be undone)")
            else:
                print("❌ Task not found!")
        else:
            print("Deletion cancelled.")
    
    def batch_toggle_interactive(self):
        """Macro command to toggle multiple tasks at once"""
        if not self.manager.tasks:
            print("\n❌ No tasks available!")
            return
        
        self.view_tasks()
        print("\nEnter task IDs to toggle (comma-separated):")
        ids_input = input("IDs: ").strip()
        
        task_ids = [tid.strip() for tid in ids_input.split(',')]
        
        if task_ids:
            self.manager.execute_macro(task_ids, f"Batch toggle {len(task_ids)} tasks")
            print(f"✅ Toggled {len(task_ids)} tasks! (Can be undone as one operation)")
        else:
            print("❌ No valid IDs provided!")
    
    def undo_interactive(self):
        if self.manager.undo():
            print("✅ Last command undone!")
        else:
            print("❌ Nothing to undo!")
    
    def redo_interactive(self):
        if self.manager.redo():
            print("✅ Command redone!")
        else:
            print("❌ Nothing to redo!")
    
    def view_command_history(self):
        history = self.manager.command_history.get_history(15)
        
        print("\n--- 📜 Command History (Last 15) ---")
        
        if not history:
            print("No commands executed yet.")
            return
        
        for i, command in enumerate(reversed(history), 1):
            print(f"{i}. {command.get_description()}")
        
        print(f"\nCan Undo: {'Yes ✅' if self.manager.can_undo() else 'No ❌'}")
        print(f"Can Redo: {'Yes ✅' if self.manager.can_redo() else 'No ❌'}")
    
    def view_statistics(self):
        stats = self.manager.get_stats()
        print("\n--- 📊 Task Statistics ---")
        print(f"Total Tasks: {stats['total']}")
        print(f"Completed: {stats['completed']}")
        print(f"Pending: {stats['pending']}")
        print(f"Completion Rate: {stats['completion_rate']:.1f}%")
    
    def run(self):
        print("Welcome to Task Manager - Phase 4!")
        print("🎯 Now with Command Pattern for Undo/Redo!")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-10): ").strip()
            
            if choice == '1':
                self.add_task_interactive()
            elif choice == '2':
                self.view_tasks()
            elif choice == '3':
                self.toggle_task_interactive()
            elif choice == '4':
                self.delete_task_interactive()
            elif choice == '5':
                self.batch_toggle_interactive()
            elif choice == '6':
                self.undo_interactive()
            elif choice == '7':
                self.redo_interactive()
            elif choice == '8':
                self.view_command_history()
            elif choice == '9':
                self.view_statistics()
            elif choice == '10':
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-10.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()