"""JSON-backed product/order tools with the same interface as database tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatabaseTools:
    products: list[dict]
    orders: list[dict]

    def search_product(self, keyword: str = "", min_price: float | None = None) -> list[dict]:
        return [p for p in self.products if keyword.lower() in p["name"].lower() and (min_price is None or p["price"] > min_price)]

    def search_order(self, order_id: str | None = None, phone: str | None = None) -> list[dict]:
        return [o for o in self.orders if (order_id is None or o["order_id"] == order_id) and (phone is None or o["phone"] == phone)]

    def product_detail(self, sku: str) -> dict | None:
        return next((p for p in self.products if p["sku"] == sku), None)
