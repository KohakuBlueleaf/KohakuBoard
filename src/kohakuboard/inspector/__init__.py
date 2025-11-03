"""KohakuBoard Inspector - GUI for viewing and managing experiment data"""

# Check if dependencies are available
try:
    import customtkinter as ctk

    INSPECTOR_AVAILABLE = True
except ImportError:
    INSPECTOR_AVAILABLE = False


def run_inspector(board_dir: str = "./kohakuboard"):
    """Run the KohakuBoard Inspector GUI

    Args:
        board_dir: Directory containing boards (default: ./kohakuboard)

    Raises:
        ImportError: If inspector dependencies not installed
    """
    if not INSPECTOR_AVAILABLE:
        raise ImportError(
            "Inspector dependencies not installed.\n"
            "Install with: pip install customtkinter\n"
            "Or: pip install kohakuboard[inspector]"
        )

    from kohakuboard.inspector.app import KohakuBoardInspector

    app = KohakuBoardInspector(board_dir)
    app.mainloop()


__all__ = ["run_inspector", "INSPECTOR_AVAILABLE"]
