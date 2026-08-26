from api import CatalogRequest, app


def main():
    route_paths = {route.path for route in app.routes}
    required_paths = {
        "/dashboard/overview",
        "/catalog/search",
        "/documents/{document_id}",
        "/comparison/options",
        "/comparison",
        "/search",
        "/chat",
        "/health",
    }
    missing = sorted(required_paths - route_paths)
    assert not missing, f"Missing API routes: {missing}"

    payload = CatalogRequest(
        query="  konut   finansmani  ",
        product_types=["KONUT_FINANSMANI", "KONUT_FINANSMANI", ""],
        bank_names=["Ziraat Katilim", "Ziraat Katilim", ""],
        min_confidence=0.8,
        sort_by="confidence",
        sort_order="desc",
        page=2,
        page_size=20,
    )
    assert payload.query == "konut finansmani"
    assert payload.product_types == ["KONUT_FINANSMANI"]
    assert payload.bank_names == ["Ziraat Katilim"]
    assert payload.page == 2
    assert payload.page_size == 20

    print("HititFinLex dashboard V2.5 API: OK")
    print("Routes:", ", ".join(sorted(required_paths)))


if __name__ == "__main__":
    main()
