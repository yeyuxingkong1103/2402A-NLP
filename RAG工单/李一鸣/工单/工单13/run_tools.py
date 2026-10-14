from tools import ToolRegistry


if __name__ == "__main__":
    registry = ToolRegistry()
    print(registry.execute("calculator", "(1200 + 300) * 0.8"))
