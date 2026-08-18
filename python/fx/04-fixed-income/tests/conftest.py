"""Shared fixtures: seeded synthetic markets per regime."""

import pytest

from fx_rates.data import build_market_state, generate_market_quotes, sample_book


@pytest.fixture(scope="session")
def quotes():
    return generate_market_quotes("normal", seed=42)


@pytest.fixture(scope="session")
def market(quotes):
    return build_market_state(quotes)


@pytest.fixture(scope="session")
def crisis_market():
    return build_market_state(generate_market_quotes("crisis", seed=42))


@pytest.fixture(scope="session")
def negative_eur_market():
    return build_market_state(generate_market_quotes("negative_eur", seed=42))


@pytest.fixture(scope="session")
def book(market):
    return sample_book(market, seed=1)
