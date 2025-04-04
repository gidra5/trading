from hypothesis import given, strategies as st


@given(st.integers(), st.integers())
def basic_decision(n, m):
