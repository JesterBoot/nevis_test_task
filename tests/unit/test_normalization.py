from search.normalization import candidate_limit, normalize_search_query


def test_company_phrase_is_compacted_without_domain_token_splitting() -> None:
    query = normalize_search_query("  Nevis Wealth  ")

    assert query.normalized == "nevis wealth"
    assert query.compact == "neviswealth"
    assert query.domain is None


def test_complete_domain_is_preserved_as_one_domain() -> None:
    query = normalize_search_query("anton.batiaev@NevisWealth.com")

    assert query.domain == "neviswealth.com"
    assert query.compact == "antonbatiaevneviswealthcom"


def test_candidate_limit_is_bounded_and_scales_past_small_limits() -> None:
    assert candidate_limit(1) == 20
    assert candidate_limit(4) == 20
    assert candidate_limit(5) == 25
    assert candidate_limit(10) == 50
    assert candidate_limit(50) == 50
