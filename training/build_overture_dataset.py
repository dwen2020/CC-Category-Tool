"""Build a merchant-categorization dataset from real Canadian businesses.

Why this exists
---------------
The previous generator (`generate_unique_merchants.py`) synthesized 82% of its rows as
"<Modifier> <Head>" names where the head noun WAS the label ("Maple Dental" -> Health).
That makes the label a function of the name, so a merchant-grouped split still leaks:
holding out "Maple Dental" teaches nothing, because "Cedar Dental" is in training and
"dental" is the whole signal. Measured consequence: a TF-IDF + LinearSVC baseline with no
pretraining scores 92.2% on that test set (99.9% on synthetic rows, 55.2% on real ones),
statistically tied with a fine-tuned DistilBERT. The benchmark was measuring the
generator, not the model.

This script instead takes real business names from Overture Maps Places and labels them
from the `basic_category` field, which is assigned independently of the name. So
"Chez Marie" is Dining and "Kaur Brothers" is Home because of what the business IS, not
because of a keyword planted in its name. Those rows are the ones that actually test
whether the encoder learned anything.

Pipeline
--------
1. `extract` - stream named Canadian places out of the Overture release into a local
   parquet. Only 3 of the 16 global shards intersect the Canada bbox and parquet
   row-group statistics prune the rest, so this reads a small fraction of the 10.5 GB.
2. `build`   - map `basic_category` onto the 12 cc_tool categories, drop non-merchants
   (lakes, schools, B2B suppliers), dedupe to one row per merchant, and render each as a
   bank-style descriptor using that merchant's REAL city and province.

Run (from the repo root, so the data/ paths below resolve correctly):
    python training/build_overture_dataset.py extract   # once, ~20s, writes data/overture_ca_places.parquet
    python training/build_overture_dataset.py build      # writes data/cc_merchants_overture.csv

Data (c) Overture Maps Foundation, ODbL 1.0.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

OVERTURE_RELEASE = "2026-07-22.0"
OVERTURE_BASE = "https://overturemaps-us-west-2.s3.amazonaws.com/"
PLACES_PREFIX = f"release/{OVERTURE_RELEASE}/theme=places/type=place/"

RAW_PARQUET = "data/overture_ca_places.parquet"
OUTPUT_CSV = "data/cc_merchants_overture.csv"

# Canada bounding box, used to prune parquet row groups before any data is transferred.
CA_BBOX = dict(xmin=-141.0, xmax=-52.0, ymin=41.0, ymax=84.0)

MIN_CONFIDENCE = 0.5      # Overture's own 0-1 certainty that the place exists
MAX_PER_CATEGORY = 8000   # cap so Services/Home do not swamp the corpus
SEED = 11

# Descriptor forms emitted per online-only merchant. Storefront merchants stay at one row
# each; digital brands get several, because for them the FORM is the whole problem - see
# render_digital. ~155 digital merchants, so even ALL of them costs ~1,400 rows on 90k, and
# every variant shares a `merchant` value so the merchant-grouped split keeps them together
# and nothing leaks across the split.
#
# 9 = every template, i.e. no sampling at all. At 5 the lottery was merely smaller rather
# than gone: eBay drew five forms but not ".CA", so EBAY.CA still scored Financial 0.74,
# and Steam missed ".COM" so STEAM.COM scored Services. There is no reason to ration
# 600 rows.
DIGITAL_VARIANTS = 9

CATEGORIES = [
    "Groceries", "Dining", "Transport", "Travel", "Shopping", "Utilities",
    "Health", "Entertainment", "Home", "Services", "Financial", "Other",
]

# ---------------------------------------------------------------------------------
# basic_category -> cc_tool category.
#
# Overture's `basic_category` is a 257-value vocabulary chosen to sit at the level of
# generality humans prefer, which makes it a far better mapping surface than raw OSM tag
# soup spread across shop=/amenity=/office=/leisure=/craft=.
#
# Ambiguous cases follow the conventions already in well_known_merchants.py rather than
# inventing new ones: gyms and fitness studios are Health (GoodLife, Fit4Less, YMCA are
# all Health there), warehouse clubs are Groceries (Costco Wholesale), fuel is Transport,
# car rental is Travel, home improvement retail is Home while trades are Services.
# ---------------------------------------------------------------------------------
BASIC_TO_CATEGORY: dict[str, str] = {}


def _assign(category: str, *basics: str) -> None:
    for b in basics:
        BASIC_TO_CATEGORY[b] = category


_assign(
    "Groceries",
    "food_and_beverage_store", "convenience_store", "farmers_market",
    "warehouse_club_store", "superstore", "market",
)
_assign(
    "Dining",
    "restaurant", "casual_eatery", "coffee_shop", "fast_food_restaurant", "cafe",
    "bar", "non_alcoholic_beverage_venue", "smoothie_juice_bar", "brewery", "winery",
    "distillery", "food_truck_stand", "food_service", "food_court",
    "alcoholic_beverage_venue", "lounge",
)
_assign(
    "Transport",
    "gas_station", "automotive_service", "vehicle_service", "vehicle_parts_store",
    "parking", "ev_charging_station", "ground_transport_facility_or_service",
    "auto_dealer", "vehicle_dealer", "taxi_or_ride_share_service",
    "public_transit_facility_or_service", "rail_facility_or_service", "ferry_service",
    "park_and_ride", "rest_stop", "train_station",
)
_assign(
    "Travel",
    "hotel", "lodging", "bed_and_breakfast", "resort", "campground", "private_lodging",
    "travel_service", "inn", "rv_park", "air_transport_facility_or_service", "airport",
)
_assign(
    "Shopping",
    "fashion_and_apparel_store", "electronics_store", "specialty_store",
    "sporting_goods_store", "second_hand_store", "arts_crafts_and_hobby_store",
    "books_music_and_video_store", "flowers_and_gifts_store", "discount_store",
    "department_store", "shopping_mall", "personal_care_and_beauty_store",
    "animal_and_pet_store", "toys_and_games_store",
    "musical_instrument_and_pro_audio_store", "office_supply_store", "kiosk",
    "shopping_service",
)
_assign(
    "Utilities",
    "telecommunications_service", "public_utility",
    "water_utility_provider", "electric_utility_provider",
    "natural_gas_utility_provider",
)
_assign(
    "Health",
    "dental_clinic", "pharmacy_and_drug_store", "wellness_service",
    "complementary_and_alternative_medicine", "physical_medicine_and_rehabilitation",
    "behavioral_or_mental_health_clinic", "medical_service", "vision_or_eye_care_clinic",
    "specialized_health_care", "hospital", "outpatient_care_facility",
    "diagnostics_imaging_or_lab_service", "primary_care_or_general_clinic",
    "gym", "fitness_studio", "sport_or_fitness_facility", "surgery",
    "specialized_medical_facility", "reproductive_perinatal_and_womens_care",
    "emergency_or_urgent_care_facility", "walk_in_clinic", "pediatric_clinic",
    "emergency_department", "urgent_care_center", "specialty_hospital",
)
_assign(
    "Entertainment",
    "movie_theater", "dance_club", "skating_rink", "golf_course", "music_venue",
    "theatre_venue", "art_gallery", "museum", "stadium_arena",
    "sport_or_recreation_club", "casino", "arcade", "comedy_club", "amusement_park",
    "amusement_attraction", "gaming_venue", "performing_arts_venue", "event_venue",
    "cultural_center", "zoo", "aquarium", "science_attraction", "planetarium",
    "adult_entertainment_venue", "skate_park", "sport_field", "sport_court",
    "sport_team", "sport_league", "festival_venue", "fairgrounds", "rodeo",
    "recreational_equipment_rental", "swimming_pool", "country_club",
    "arts_and_crafts_space", "social_club", "ticket_office_or_booth",
    "animal_attraction", "makerspace", "class_venue", "marina",
)
_assign(
    "Home",
    "hardware_home_and_garden_store", "storage_facility", "housing_or_property_service",
)
_assign(
    "Services",
    "home_service", "personal_or_beauty_service", "professional_service",
    "attorney_or_law_firm", "legal_service", "printing_service", "laundry_service",
    "animal_or_pet_service", "shipping_or_delivery_service", "technical_service",
    "event_or_party_service", "family_service", "rental_service", "media_service",
    "tutoring_service", "building_or_construction_service", "security_service",
    "print_media_service", "psychic_advising", "astrological_advising",
    "spiritual_advising", "environmental_or_ecological_service", "agricultural_service",
    "educational_service", "specialty_school", "real_estate_service",
    # NOTE: animal_or_pet_service stays here on purpose. Its leaves are veterinarian
    # (3,613), pet_groomer (3,184), pet_boarding and dog_trainer - all genuinely paid on a
    # personal card. With no Pets category to hold them, Services is the right bucket.
)
_assign("Financial", "financial_service", "bank_or_credit_union", "atm")
_assign(
    "Other",
    "social_or_community_service", "government_office", "civic_organization",
    "community_center", "library", "religious_organization",
    "christian_place_of_worship", "muslim_place_of_worship",
    "buddhist_place_of_worship", "hindu_place_of_worship", "jewish_place_of_worship",
    "place_of_worship", "youth_organization", "labor_union", "political_organization",
    "food_bank", "courthouse", "embassy", "government_department", "police_station",
    "fire_station", "jail_or_prison", "military_site", "school_district_office",
    "civic_center", "radio_station", "television_station", "research_institute",
    "monument",
)

# ---------------------------------------------------------------------------------
# Refinements keyed on the FINER taxonomy leaf, applied after BASIC_TO_CATEGORY.
#
# Overture's categories form a tree: `tax_hierarchy` is the full path, `tax_primary` is
# the leaf, and `basic_category` is a mid-level node. Mapping the mid-level node alone is
# right for most branches (`restaurant` has 139 leaves and they are all Dining) but wrong
# where a parent mixes consumer-spend types. Measured leaf counts are in the comments.
#
# Keys are (basic_category, tax_primary) PAIRS, not bare leaves: a leaf name is not
# globally unique. `rental_service` appears under both the `rental_service` parent and
# `vehicle_service`, and `automotive_service` under three parents, so keying on the leaf
# alone would silently rewrite branches this is not aiming at.
TAX_OVERRIDES: dict[tuple[str, str], str] = {
    # Overture files municipal waste under public_utility, which is defensible in civic
    # terms and wrong for a card statement: you pay a private junk-removal or septic
    # contractor, not a monthly utility bill. This is 88% of the public_utility branch by
    # tax_primary (1,513 + 419 vs 258 genuine utilities) and it was contaminating 21% of
    # the Utilities class - the smallest one, so it hurt proportionally more.
    ("public_utility", "garbage_collection_service"): "Services",
    ("public_utility", "septic_service"): "Services",

    # Car and RV rental. The convention above already says "car rental is Travel", and the
    # curated catalog labels Enterprise / Alamo / National / Budget as Travel - but the
    # Overture rental_service branch was overriding all of them to Services, which showed
    # up as a cross-source label conflict on exactly those chains.
    # Truck and trailer rental deliberately stay Services: a U-Haul is a moving purchase,
    # not a trip.
    ("rental_service", "car_rental_service"): "Travel",      # 2,408
    ("rental_service", "rv_rental_service"): "Travel",        # 71

    # An ISP is a utility bill, not a technical service call.
    ("technical_service", "internet_service_provider"): "Utilities",   # 181
}

# Explicitly NOT merchants: you cannot put a lake on a credit card. Listed rather than
# left to fall through, so a new Overture category shows up as "unmapped" in the report
# instead of being silently dropped.
EXCLUDED = {
    # natural and civic features
    "lake", "river", "mountain", "beach", "park", "national_park", "nature_reserve",
    "recreational_trail_or_path", "island", "waterfall", "garden", "forest", "canal",
    "canyon", "hot_springs", "land_feature", "built_feature", "bridge", "pier",
    "lighthouse", "public_plaza", "public_fountain", "public_restroom", "playground",
    "dog_park", "cemetery", "sculpture_statue", "street_art", "memorial_site",
    "castle", "fort", "rural_attraction",
    # not consumer-facing
    "farm", "manufacturer", "industrial_facility_or_service", "supplier_or_distributor",
    "wholesaler", "corporate_or_business_office", "b2b_service",
    "b2b_transportation_and_storage_service", "b2b_office_and_professional_service",
    "b2b_industrial_and_machine_service", "b2b_science_and_technology_service",
    # Oilfield and industrial-energy contractors (Schlumberger et al). Not a consumer
    # utility biller, despite the name -- keeps Utilities meaning "hydro/telecom/gas".
    "b2b_energy_and_utility_service",
    # schooling and residential - rarely a card descriptor
    "elementary_school", "high_school", "middle_school", "preschool",
    "place_of_learning", "college_university", "educational_facility",
    "apartment", "condominium", "senior_living_facility", "campus_building",
    # Heritage plaques, cairns and markers. 27,212 records, and the overwhelming majority
    # charge nothing - there is no transaction to categorize. The ones that DO sell
    # admission are museums, and `museum` is already mapped to Entertainment.
    "historic_site",
    # Web, graphic and sign-making firms plus architects. Overture files all six leaves
    # under design_service; none of them is a purchase that shows up on a personal card.
    "design_service",
}

# Leaf-level exclusions, keyed on (basic_category, tax_primary) like TAX_OVERRIDES.
#
# `professional_service` is a 28,009-record branch that mixes consumer purchases with
# pure B2B. Agencies bill other companies, not cardholders, so training on them teaches
# the model nothing about statements while diluting Services - which is already the
# weakest class in the report (recall 0.39).
#
# Kept deliberately: cleaning_service (863) and bookkeeper (225) are things an individual
# pays for; so are career_counseling (155), translation_service (97) and
# immigration_assistance_service (18). Commercial cleaning is excluded separately from
# residential - janitorial/office/industrial bill a building, not a person.
TAX_EXCLUDED: set[tuple[str, str]] = {
    ("professional_service", leaf) for leaf in (
        # advertising / marketing / PR agencies
        "advertising_agency", "marketing_agency", "internet_marketing_service",
        "e_commerce_service", "public_relations", "social_media_agency",
        "merchandising_service", "copywriting_service", "writing_service",
        "editorial_service",
        # staffing and consulting
        "employment_agency", "temp_agency", "talent_agency", "business_consulting",
        "food_and_beverage_consultant", "certification_agency",
        # commercial-only back-office and facilities
        "janitorial_service", "office_cleaning", "industrial_cleaning_service",
        "payroll_service", "billing_service", "shredding_service",
        "bank_equipment_service",
        # 263 records, and the sample is not coworking at all - it contains an ADT
        # branch, a Sylvan Learning Center and a Starbucks regional support office.
        "coworking_space",
        # The parent-named leaf, and the largest in the branch at 13,541 records. It is a
        # dumping ground rather than a category: the sample holds Gill-Power Hobby Farm,
        # Froghome Farm, Cherry Point Marina, Comsense Kitchen Cabinets, Pickles' Pantry
        # and Living Stones Trucking. Several contradict this file's own mapping outright
        # (a marina is Entertainment, a pantry reads as Groceries), so these rows teach
        # Services - already the weakest class at recall 0.39 - a near-random label.
        "professional_service",
    )
}


# ---------------------------------------------------------------------------------
# Step 1: extract
# ---------------------------------------------------------------------------------

def _connect(duckdb):
    con = duckdb.connect()
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
    proxy = re.sub(r"^https?://", "", proxy).rstrip("/")
    if proxy:
        # DuckDB needs host:port with no scheme, and needs it set before httpfs loads
        # or the extension download itself fails behind a corporate proxy.
        con.execute(f"SET http_proxy='{proxy}';")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false; SET preserve_insertion_order=false;")
    return con


def _list_place_files(con) -> list[str]:
    """Shard keys that actually intersect Canada, found via row-group statistics."""
    listing = con.execute(
        f"SELECT file FROM glob('{OVERTURE_BASE}{PLACES_PREFIX}*.parquet')"
    ).fetchall()
    files = [r[0] for r in listing]
    hits = []
    for f in files:
        n = con.execute(
            f"""SELECT count(*) FROM read_parquet('{f}')
                WHERE bbox.xmin BETWEEN {CA_BBOX['xmin']} AND {CA_BBOX['xmax']}
                  AND bbox.ymin BETWEEN {CA_BBOX['ymin']} AND {CA_BBOX['ymax']}"""
        ).fetchone()[0]
        print(f"  {os.path.basename(f)[:16]}  bbox rows: {n:>9}")
        if n:
            hits.append(f)
    return hits


def extract() -> None:
    try:
        import duckdb
    except ImportError:
        sys.exit("pip install duckdb")

    con = _connect(duckdb)
    print(f"scanning {OVERTURE_RELEASE} places shards for Canada...")
    files = _list_place_files(con)
    if not files:
        sys.exit("no shards intersect the Canada bbox - has the release path changed?")
    print(f"{len(files)} shard(s) contain Canadian data")

    # `taxonomy.primary` is what TAX_OVERRIDES keys on, so it must stay in this SELECT.
    # Note the checked-in overture_ca_places.parquet also carries `tax_hierarchy` and
    # `cat_primary` from an earlier version of this query; both are audit conveniences
    # (the full category path, and the older raw category field) and re-running extract
    # drops them. Nothing in build() depends on either.
    srcs = ", ".join(f"'{f}'" for f in files)
    con.execute(f"""
        COPY (
          SELECT names.primary          AS name,
                 basic_category         AS basic_category,
                 taxonomy.primary       AS tax_primary,
                 addresses[1].locality  AS city,
                 addresses[1].region    AS region,
                 brand.names.primary    AS brand,
                 operating_status       AS operating_status,
                 confidence             AS confidence
          FROM read_parquet([{srcs}])
          WHERE bbox.xmin BETWEEN {CA_BBOX['xmin']} AND {CA_BBOX['xmax']}
            AND bbox.ymin BETWEEN {CA_BBOX['ymin']} AND {CA_BBOX['ymax']}
            AND addresses[1].country = 'CA'
            AND names.primary IS NOT NULL
        ) TO '{RAW_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{RAW_PARQUET}')").fetchone()[0]
    print(f"wrote {RAW_PARQUET}: {n:,} named Canadian places")


# ---------------------------------------------------------------------------------
# Step 2: descriptor rendering
# ---------------------------------------------------------------------------------

# Names that are placeholders, fragments, or too generic to carry a label.
_JUNK = re.compile(
    r"^(unnamed|unknown|n/?a|none|null|test|tbd|vacant|closed|private)\b", re.I)
_HAS_LETTER = re.compile(r"[A-Za-z]")


def usable_name(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 42:
        return False
    if not _HAS_LETTER.search(name):
        return False
    if _JUNK.match(name.strip()):
        return False
    # Overture occasionally carries a bare street address as a name.
    if re.match(r"^\d+\s+\w+\s+(st|street|ave|avenue|rd|road|blvd)\b", name, re.I):
        return False
    return True


# Overture names sometimes carry a branch qualifier that a bank would never print.
_BRANCH_TAIL = re.compile(
    r"\s*[-–(]\s*(?:#?\d+|[^-–()]{0,24}(?:branch|location|store|plaza|mall|centre|center"
    r"|square|and|&)[^-–()]{0,24})\s*\)?\s*$", re.I)


def clean_business_name(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = _BRANCH_TAIL.sub("", name).strip(" -–(),")
    return name or prev


def render_descriptor(name: str, city: str | None, region: str | None,
                      is_chain: bool, rng: random.Random) -> str:
    """Render a merchant the way a bank prints it, using its real city and province.

    Nationally-known brands run their own merchant accounts, so they never carry a
    Square/Toast aggregator prefix -- "SQ *TIM HORTONS" is not a thing. Those prefixes
    belong to independents, which is exactly what `brand IS NULL` identifies.
    """
    up = name.upper()
    cityu = (city or "").upper()
    prov = (region or "").upper()
    store = rng.randint(1, 9999)

    forms = [f"{up}"]
    if cityu and prov:
        forms += [
            f"{up} #{store} {cityu} {prov}",
            f"{up} {cityu} {prov}",
            f"{up} {store} {cityu}",
        ]
    if prov:
        forms += [f"{up} {prov}"]
    forms += [f"{up}#{store}", f"{up} STORE #{store}", f"POS PURCHASE {up}"]
    if not is_chain:
        forms += [f"SQ *{up}", f"TST* {up}", f"SP * {up}"]

    # Banks clip the descriptor field around 32 characters.
    return rng.choice(forms)[:32].strip()


# ---------------------------------------------------------------------------------
# Curated catalog: chain canonicalization + the non-POI population
# ---------------------------------------------------------------------------------

def load_curated() -> tuple[dict, set]:
    """Return (name -> category, digital-only names) from the hand-curated catalogs.

    Overture only populates `brand` for about 9% of rows, so a chain like Loblaws shows
    up as ~18 slightly different names that a merchant-grouped split would treat as 18
    independent merchants -- putting the same business on both sides of the split. The
    curated brand list is used to canonicalize those back to one merchant.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    curated: dict[str, str] = {}
    digital: set = set()
    try:
        from generate_merchant_dataset import MERCHANTS
        for cat, names in MERCHANTS.items():
            for n in names:
                curated.setdefault(n, cat)
    except ImportError:
        print("  (generate_merchant_dataset not importable - skipping curated brands)")
    try:
        from well_known_merchants import EXTRA_MERCHANTS, LABEL_FIXES, NAME_FIXES
        for cat, names in EXTRA_MERCHANTS.items():
            for n in names:
                curated.setdefault(n, cat)
        for bad, good in NAME_FIXES.items():
            if bad in curated:
                curated[good] = curated.pop(bad)
        for n, cat in LABEL_FIXES.items():
            if n in curated:
                curated[n] = cat
    except ImportError:
        print("  (well_known_merchants not importable)")
    try:
        from generate_unique_merchants import DIGITAL
        digital = set(DIGITAL)
    except ImportError:
        pass
    return curated, digital


_CODE = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"

# Only used for curated merchants, which carry no Overture address of their own. Real
# Overture rows always render with the merchant's actual city and province.
_FALLBACK_CITIES = [
    ("Toronto", "ON"), ("Montreal", "QC"), ("Vancouver", "BC"), ("Calgary", "AB"),
    ("Edmonton", "AB"), ("Ottawa", "ON"), ("Winnipeg", "MB"), ("Quebec City", "QC"),
    ("Hamilton", "ON"), ("Halifax", "NS"), ("Victoria", "BC"), ("Saskatoon", "SK"),
    ("Regina", "SK"), ("Mississauga", "ON"), ("Brampton", "ON"), ("Surrey", "BC"),
    ("Laval", "QC"), ("Markham", "ON"), ("Gatineau", "QC"), ("Burnaby", "BC"),
]


def render_digital(name: str, rng: random.Random, n: int = 1) -> list[str]:
    """Descriptor forms for an online-only merchant, as a list of up to `n` distinct ones.

    Online merchants never print a store number or a city - they print a domain, a support
    number, or a processor-plus-code.

    Returns SEVERAL forms rather than one. One row per merchant meant every online brand
    drew a single template out of ten and the model only ever saw that draw. Measured
    consequences: Netflix drew `NETFLIX.CA`, so `NETFLIX.COM` was off-distribution and fell
    through to the ISP character n-grams ("net", ".com") for a 0.88 Utilities prediction;
    eBay drew a phone number, so `EBAY.CA` read as Financial at 0.64. These are exactly the
    brands a real statement is full of, so they are the ones that need format coverage.

    The old `{NAME} INTERNET` template is deliberately gone. No bank prints "APPLE TV
    INTERNET" or "DISCORD NITRO INTERNET", and it was stamping the single strongest
    Utilities token onto Entertainment and Services brands - 19 rows of pure noise.
    """
    up = name.upper()

    def _code() -> str:
        return "".join(rng.choice(_CODE) for _ in range(rng.randint(4, 6)))

    def _phone() -> str:
        return f"{rng.randint(800, 899)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"

    forms = [
        f"{up}.COM", f"{up}.CA", f"{up}.COM/BILL", f"PAYPAL *{up}", f"GOOGLE *{up}",
        f"{up}*{_code()}", f"{up} {_phone()}", f"SP * {up}", f"{up}",
    ]
    rng.shuffle(forms)

    out: list[str] = []
    seen: set[str] = set()
    for form in forms:
        form = form[:32].strip()
        if form and form not in seen:
            seen.add(form)
            out.append(form)
        if len(out) >= n:
            break
    return out


def build_brand_matcher(curated: dict) -> list:
    """Longest-first list of (lowercased brand, canonical brand) for prefix matching."""
    pairs = [(n.lower(), n) for n in curated if len(n) >= 4]
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


# Brands whose name is an ordinary word or a surname. Prefix matching is right for
# "SHELL #10214" or "SOBEYS #3189" - those are real locations of a chain - but wrong here,
# because the prefix also pulls in unrelated businesses that merely start with the word.
# Measured absorption before this list existed: "Amazon" took 500 distinct names (Amazon
# Auto Sales Ltd, Amazon Bar And Grill, Amazon Construction Group, Amazon Driving School,
# even Amazon Creek), and "Rogers" took 162 (ROGERS ROAD, Rogers & Company, ROGERS tv Grey
# County) - which mattered doubly once Rogers was routed to Utilities. For these, only an
# exact name match counts.
#
# Other common-word brands with the same exposure, left alone for now because they predate
# this and changing them shifts existing labels: Apple (196 names: APPLE HILL PO, APPLE
# Program, Apple & Pears Group Daycare), Dominion (270), Maxi (231), Square (138).
EXACT_ONLY_BRANDS = {"amazon", "rogers"}


def canonicalize(name: str, matcher: list) -> str:
    """If a name starts with a known brand, return the brand. 'Loblaws City Market' and
    'Loblaws #1234' both collapse to 'Loblaws' - unless the brand is in
    EXACT_ONLY_BRANDS, where only an exact match counts."""
    low = name.lower()
    for brand_low, brand in matcher:
        if low == brand_low:
            return brand
        if brand_low not in EXACT_ONLY_BRANDS and low.startswith(brand_low + " "):
            return brand
    return name


# ---------------------------------------------------------------------------------
# Step 2: build
# ---------------------------------------------------------------------------------

def build() -> None:
    import pandas as pd

    if not os.path.exists(RAW_PARQUET):
        sys.exit(f"{RAW_PARQUET} not found - run `python {sys.argv[0]} extract` first")

    rng = random.Random(SEED)
    df = pd.read_parquet(RAW_PARQUET)
    print(f"loaded {len(df):,} places")

    # Quality gates.
    df = df[df["confidence"] >= MIN_CONFIDENCE]
    df = df[df["operating_status"].isna() | (df["operating_status"] != "permanently_closed")]
    print(f"  after confidence>={MIN_CONFIDENCE} and open-status: {len(df):,}")

    # Report any Overture category this script has never seen, so the mapping can be
    # kept current instead of silently losing rows to a schema change.
    known = set(BASIC_TO_CATEGORY) | EXCLUDED
    seen = set(df["basic_category"].dropna().unique())
    unmapped = seen - known
    if unmapped:
        counts = df[df["basic_category"].isin(unmapped)]["basic_category"].value_counts()
        print(f"  WARNING: {len(unmapped)} unmapped basic_category values "
              f"({int(counts.sum()):,} rows) - add them to the mapping:")
        for k, v in counts.head(20).items():
            print(f"    {v:>7}  {k}")

    df = df[df["basic_category"].isin(BASIC_TO_CATEGORY)].copy()

    # Leaf-level drops, before the mapping so the counts below reflect what survives.
    if TAX_EXCLUDED and "tax_primary" in df.columns:
        pair = list(zip(df["basic_category"], df["tax_primary"]))
        drop = pd.Series([p in TAX_EXCLUDED for p in pair], index=df.index)
        if drop.any():
            gone = (df.loc[drop].groupby(["basic_category", "tax_primary"])
                      .size().sort_values(ascending=False))
            print(f"  leaf-level exclusions: -{int(drop.sum()):,} rows")
            for (b, t), n in gone.head(10).items():
                print(f"    {n:>6,}  {b} > {t}")
            if len(gone) > 10:
                print(f"    {int(gone.iloc[10:].sum()):>6,}  (+{len(gone) - 10} smaller leaves)")
            df = df[~drop]
        stale = TAX_EXCLUDED - set(pair)
        for key in sorted(stale):
            print(f"  WARNING: TAX_EXCLUDED key matched 0 rows: {key}")

    df["category"] = df["basic_category"].map(BASIC_TO_CATEGORY)
    print(f"  after mapping to the 12 categories: {len(df):,}")

    # Refine with the finer taxonomy leaf where the mid-level node rolls up too coarsely.
    if "tax_primary" not in df.columns:
        print("  WARNING: no tax_primary column - re-run `extract` to apply TAX_OVERRIDES")
    else:
        override = pd.Series(
            list(TAX_OVERRIDES.values()),
            index=pd.MultiIndex.from_tuples(TAX_OVERRIDES.keys()),
            dtype="object",
        )
        pair = pd.MultiIndex.from_arrays([df["basic_category"], df["tax_primary"]])
        refined = pd.Series(override.reindex(pair).to_numpy(), index=df.index)
        hit = refined.notna()
        if hit.any():
            moved = (df.loc[hit.to_numpy()]
                       .assign(to=refined[hit])
                       .groupby(["basic_category", "tax_primary", "category", "to"])
                       .size().sort_values(ascending=False))
            print(f"  tax_primary refinements: {int(hit.sum()):,} rows")
            for (b, t, was, now), n in moved.items():
                print(f"    {n:>6,}  {b} > {t}:  {was} -> {now}")
            df["category"] = refined.fillna(df["category"])
        # A stale key means the release renamed a leaf; surface it instead of no-oping.
        seen = set(zip(df["basic_category"], df["tax_primary"]))
        for key in TAX_OVERRIDES:
            if key not in seen:
                print(f"  WARNING: TAX_OVERRIDES key matched 0 rows: {key}")

    # Clean names and drop junk.
    df["merchant"] = df["name"].astype(str).map(clean_business_name)
    df = df[df["merchant"].map(usable_name)]
    print(f"  after name cleanup: {len(df):,}")

    # Collapse chains to one merchant. A brand with 3,000 locations is ONE merchant for
    # split purposes, and letting all 3,000 through would swamp the independents.
    curated, digital = load_curated()
    print(f"  curated catalog: {len(curated):,} brands, {len(digital):,} online-only")
    matcher = build_brand_matcher(curated)

    # Overture's own brand field first, then curated prefix matching for the ~91% of
    # rows where Overture leaves brand NULL.
    df["merchant"] = df["brand"].fillna(df["merchant"])
    before = df["merchant"].nunique()
    df["merchant"] = df["merchant"].map(lambda n: canonicalize(n, matcher))
    print(f"  chain canonicalization: {before:,} -> {df['merchant'].nunique():,} names")
    df["group_key"] = df["merchant"].str.lower().str.strip()
    df["is_chain"] = df["brand"].notna() | df["merchant"].isin(curated)

    # Curated labels win, unconditionally.
    #
    # This used to be enforced only by the `extra` loop below, which adds a curated brand
    # ONLY if its name is absent from df - so any brand Overture also knows about kept the
    # Overture label. That silently mislabelled the chains that matter most: 347 Overture
    # records are named "Rogers" and 334 are electronics_store > mobile_phone_store, so
    # Rogers came out as Shopping rather than Utilities. Bell Canada landed on Home,
    # Spotify on Home, Circle K on Groceries, Kayak on Shopping.
    #
    # Overture is not wrong about those places - a Rogers phone store IS a shopping
    # destination. But this dataset models STATEMENT DESCRIPTORS, and "ROGERS" on a
    # statement is the monthly bill, not a handset purchase. Where the two readings
    # disagree the curated catalog is the one that matches the task, so it wins.
    #
    # Applied after canonicalization so "Rogers Wireless" -> "Rogers" inherits it, and
    # before the cap so each category samples from a corrected pool.
    # ...with one exception, where the CATALOG is the outlier instead of Overture.
    #
    # The catalog files auto-repair chains as Services (NAPA AutoPro, CARSTAR, Fix Auto,
    # OK Tire, Kal Tire, Tirecraft, Midas, Jiffy Lube, Boyd Autobody, 26 brands / 3,934
    # records). But BASIC_TO_CATEGORY maps automotive_service -> Transport, and 72,416
    # independent auto shops arrive as Transport. Letting the chains win would split ONE
    # business type by chain-ness - OK Tire as Services, the independent tire shop next
    # door as Transport - which is precisely the contradiction TAX_OVERRIDES exists to
    # remove. The majority convention wins, and grouping car costs under Transport is what
    # a spending tracker wants anyway.
    CURATED_LOSES = {("Transport", "Services")}

    cur_map = {n.lower(): c for n, c in curated.items() if c in CATEGORIES}
    cur_cat = df["group_key"].map(cur_map)

    # Decide suppression per BRAND, not per record. Per-record was not enough: a chain
    # whose locations carry more than one Overture category gets the rule applied to some
    # rows and not others, and then the brand's final label is whichever location happens
    # to win the highest-confidence dedupe below. Kal Tire flipped to Services that way
    # while the other 13 auto chains stayed Transport.
    per_record = pd.Series(
        [(a, b) in CURATED_LOSES for a, b in zip(df["category"], cur_cat)],
        index=df.index,
    )
    losing = df["group_key"].isin(set(df.loc[per_record, "group_key"]))

    # brand -> the label it keeps. Used twice: to pin every one of the brand's records to
    # that label here, and again by the `extra` loop at the end, which re-adds any curated
    # brand missing from df - a suppressed brand goes missing whenever the per-category cap
    # samples it out, and would otherwise be handed back the curated label just suppressed.
    suppressed_label: dict[str, str] = {}
    if losing.any():
        suppressed_label = dict(zip(df.loc[per_record, "group_key"],
                                    df.loc[per_record, "category"]))
        held = df.loc[losing, "merchant"].nunique()
        print(f"  curated label SUPPRESSED on {int(losing.sum()):,} records "
              f"({held} brands) - see CURATED_LOSES")
        # Pin the whole brand, so dedupe cannot reintroduce a stray label.
        df["category"] = df["group_key"].map(suppressed_label).fillna(df["category"])
    conflict = cur_cat.notna() & (cur_cat != df["category"]) & ~losing
    if conflict.any():
        changed = (df.loc[conflict]
                     .assign(to=cur_cat[conflict])
                     .groupby(["merchant", "category", "to"]).size()
                     .reset_index(name="n").sort_values("n", ascending=False))
        print(f"  curated label wins on {conflict.sum():,} records "
              f"({changed['merchant'].nunique():,} distinct brands):")
        for r in changed.head(20).itertuples():
            print(f"    {r.n:>5,}  {r.merchant:28s} {r.category:14s} -> {r.to}")
        if len(changed) > 20:
            print(f"    {'':>5}  (+{len(changed) - 20} more brand/from/to entries)")
        # Assign through `conflict`, NOT through cur_cat alone. A bare
        # `cur_cat.fillna(df["category"])` writes the curated label onto every curated
        # row including the CURATED_LOSES ones, silently undoing the pin above.
        df["category"] = cur_cat.where(conflict).fillna(df["category"])

    # One row per merchant, keeping the highest-confidence record (which also carries the
    # most trustworthy city/region).
    df = df.sort_values("confidence", ascending=False)
    df = df.drop_duplicates(subset=["group_key"], keep="first")
    print(f"  after dedupe to one row per merchant: {len(df):,}")

    # A name that maps to two categories across sources is ambiguous by construction.
    dupe_name = df.groupby(df["merchant"].str.lower())["category"].transform("nunique")
    df = df[dupe_name == 1]
    print(f"  after dropping cross-category name collisions: {len(df):,}")

    # Cap the dominant categories.
    parts = []
    for cat, sub in df.groupby("category"):
        if len(sub) > MAX_PER_CATEGORY:
            sub = sub.sample(MAX_PER_CATEGORY, random_state=SEED)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)

    # Hold online-only merchants back from storefront rendering, and route on the DIGITAL
    # set rather than on which code path a merchant arrived by. Overture does hold POI
    # records for some online brands, and those were being given store numbers and
    # provinces: "CRAVE STORE #2304", "SPOTIFY ON", "UBER AB". No statement prints those.
    is_digital = df["merchant"].str.lower().str.strip().isin(digital)
    digital_pois = df[is_digital]
    df = df[~is_digital].copy()
    if len(digital_pois):
        print(f"  online-only brands held back from storefront rendering: "
              f"{digital_pois['merchant'].nunique():,}")

    # Render one descriptor per storefront merchant.
    df["transaction_description"] = [
        render_descriptor(r.merchant, r.city, r.region, r.is_chain, rng)
        for r in df.itertuples()
    ]

    # Add the curated storefront merchants Overture happens to be missing.
    have = set(df["merchant"].str.lower())
    extra = []
    for name, cat in curated.items():
        if name.lower() in have or cat not in CATEGORIES or name.lower() in digital:
            continue          # digital brands are handled together, below
        # Honour CURATED_LOSES here too, so a brand's label does not depend on whether
        # the cap happened to sample its Overture row out.
        cat = suppressed_label.get(name.lower(), cat)
        city, region = rng.choice(_FALLBACK_CITIES)
        extra.append({"transaction_description": render_descriptor(name, city, region, True, rng),
                      "category": cat, "merchant": name, "is_chain": True})
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        print(f"  added {len(extra):,} curated storefront merchants")

    # One row per storefront merchant. Done BEFORE the digital rows are appended, because
    # digital brands are deliberately allowed several rows each.
    df = df[df["transaction_description"].str.len() >= 3]
    df["_mkey"] = df["merchant"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["_mkey"])

    # ---- online-only merchants: several descriptor forms each ------------------------
    # Both sources in one place: brands Overture knows about (held back above) and the
    # curated subscriptions/streaming/e-commerce that no POI database can contain. These
    # are the Zipf head of a real statement, so they are the rows worth spending on.
    digital_names: dict[str, str] = {}
    for r in digital_pois.itertuples():
        digital_names.setdefault(r.merchant, r.category)
    for name, cat in curated.items():
        if cat in CATEGORIES and name.lower() in digital:
            digital_names.setdefault(name, suppressed_label.get(name.lower(), cat))

    dig_rows = []
    for name, cat in sorted(digital_names.items()):
        for desc in render_digital(name, rng, n=DIGITAL_VARIANTS):
            dig_rows.append({"transaction_description": desc, "category": cat,
                             "merchant": name, "is_chain": True})
    if dig_rows:
        df = pd.concat([df, pd.DataFrame(dig_rows)], ignore_index=True)
        print(f"  added {len(dig_rows):,} rows for {len(digital_names):,} online-only "
              f"merchants ({DIGITAL_VARIANTS} descriptor forms each)")

    df = df[df["transaction_description"].str.len() >= 3]
    df = df.drop_duplicates(subset=["transaction_description"])

    out = df[["transaction_description", "category", "merchant"]]
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"wrote {OUTPUT_CSV}")
    print(f"  rows              : {len(out):,}")
    print(f"  unique merchants  : {out['merchant'].nunique():,}")
    print(f"  unique descriptors: {out['transaction_description'].nunique():,}")
    print(f"  chains            : {int(df['is_chain'].sum()):,}")
    print(f"  independents      : {int((~df['is_chain']).sum()):,}")
    print()
    print(out["category"].value_counts().to_string())
    print()
    print(out.head(25).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["extract", "build"])
    args = ap.parse_args()
    {"extract": extract, "build": build}[args.step]()


if __name__ == "__main__":
    main()
