"""Canonical spending-category list. Single source of truth -- reference, do not
duplicate.

These apply to `purchase` rows only. Payments, refunds, fees, and interest are
identified by `TransactionRow.transaction_type` and are never assigned a spending
category, so they are deliberately absent from this list.

Keep the list tight: merging categories later is trivial, splitting one
retroactively is not. "Other" is the closed-set escape hatch -- the categorizer
must pick from this list and never invent a new label.
"""

CATEGORIES = [
    "Groceries",       # supermarkets, grocery stores
    "Dining",          # restaurants, cafes, coffee, bars, fast food, food delivery
    "Transport",       # gas, transit, rideshare, parking, tolls, taxis
    "Travel",          # flights, hotels, car rental, airlines, booking sites
    "Shopping",        # retail, clothing, electronics, general merchandise
    "Utilities",       # phone, internet, hydro, gas bill, water
    "Health",          # pharmacy, doctors, dental, fitness/gym
    "Entertainment",   # streaming, games, movies, events, subscriptions
    "Home",            # furniture, hardware, home improvement, household supplies
    "Services",        # personal/professional services, repairs, salons, memberships
    "Financial",       # insurance, transfers, financial products
    "Other",           # genuine catch-all when nothing else fits
]

# Fast membership checks / validation.
CATEGORY_SET = frozenset(CATEGORIES)

# Sentinel for a row that has not been categorized (non-purchase rows, or a
# purchase awaiting categorization).
UNCATEGORIZED = None
