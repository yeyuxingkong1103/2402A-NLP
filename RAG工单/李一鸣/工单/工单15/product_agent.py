"""Product specialist agent with explicit role registration and tools."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Product:
    sku: str
    name: str
    price: float
    stock: int
    category: str


class ProductAgent:
    role = "product_agent"

    def __init__(self, model, products: list[Product] | None = None):
        self.model = model
        self.products = products or []
        self.tools: dict[str, Callable] = {
            "search_product": self.search_product,
            "product_detail": self.product_detail,
        }

    def search_product(self, keyword: str) -> list[dict]:
        return [p.__dict__ for p in self.products if keyword.lower() in p.name.lower() or keyword.lower() in p.category.lower()]

    def product_detail(self, sku: str) -> dict | None:
        return next((p.__dict__ for p in self.products if p.sku == sku), None)

    def handle(self, question: str) -> str:
        return self.model.invoke(question, tools=list(self.tools))


def register_agent(registry: dict, agent: ProductAgent) -> None:
    registry[agent.role] = agent
