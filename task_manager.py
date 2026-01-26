"""
Task Manager - Phase 1: Basic CRUD Operations
A simple CLI task manager to learn programming patterns
"""

from datetime import datetime
from typing import List, Optional


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
    
    def display_full(self) -> str:
        """Display full task details"""
        status = "Completed" if self.completed else "Pending"
        result = f"\n{'='*50}\n"
        result += f"ID: {self.id}\n"
        result += f"Title: {self.title}\n"
        result += f"Description: {self.description or 'No description'}\n"
        result += f"Status: {status}\n"
        result += f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"{'='*50}\n"
        return result


class TaskManager:
    """Manages the collection of tasks - CRUD operations"""
    
    def __init__(self):
        self.tasks: List[Task] = []
    
    # CREATE
    def add_task(self, title: str, description: str = "") -> Task:
        """Add a new task to the manager"""
        if not title.strip():
            raise ValueError("Task title cannot be empty")
        
        task = Task(title.strip(), description.strip())
        self.tasks.append(task)
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
        """Toggle completion status of a task"""
        task = self.get_task_by_id(task_id)
        if task:
            task.toggle_completion()
            return True
        return False
    
    # DELETE
    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID"""
        task = self.get_task_by_id(task_id)
        if task:
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


class TaskManagerCLI:
    """Command-line interface for the Task Manager"""
    
    def __init__(self):
        self.manager = TaskManager()
        self.running = True
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("📋 TASK MANAGER - Phase 1")
        print("="*50)
        print("1. Add Task")
        print("2. View All Tasks")
        print("3. View Pending Tasks")
        print("4. View Completed Tasks")
        print("5. Complete/Uncomplete Task")
        print("6. Delete Task")
        print("7. View Statistics")
        print("8. Exit")
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
        
        print("\n--- 📊 Statistics ---")
        print(f"Total Tasks: {stats['total']}")
        print(f"Completed: {stats['completed']}")
        print(f"Pending: {stats['pending']}")
        print(f"Completion Rate: {stats['completion_rate']:.1f}%")
    
    def run(self):
        """Main application loop"""
        print("Welcome to Task Manager!")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (1-8): ").strip()
            
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
                print("\n👋 Goodbye! Thanks for using Task Manager!")
                self.running = False
            else:
                print("\n❌ Invalid choice! Please enter 1-8.")


if __name__ == "__main__":
    app = TaskManagerCLI()
    app.run()