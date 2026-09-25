"""Country-aware normalization lexicons (general domain knowledge, no lookups).

Token maps are applied to single lowercase, accent-folded tokens; component
maps are applied to a whole comma-separated address component.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Names (all countries)

NAME_TOKEN_MAP = {
    "pvt": "private", "pte": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited", "limted": "limited",
    "co": "company", "cos": "company", "cie": "company",
    "corp": "corporation", "incorporation": "incorporated", "inc": "incorporated",
    "intl": "international",
    "mfg": "manufacturing", "mfrs": "manufacturers",
    "bros": "brothers", "assoc": "associates", "assn": "association",
    "svcs": "services", "svc": "service", "servs": "services",
    "grp": "group", "tech": "technologies", "technology": "technologies",
    "inds": "industries", "industry": "industries",
    "entp": "enterprises", "enterprise": "enterprises",
    "mgmt": "management", "dev": "development", "devt": "development",
    "natl": "national", "univ": "university",
    "ets": "etablissements", "etab": "etablissements",
    # honorific spellings -> one canonical form
    "shri": "sri", "shree": "sri", "sree": "sri", "shreee": "sri", "srii": "sri",
    "smt": "smt",
}

# Removed from the "core" name used for blocking: legal forms, honorifics,
# trade-name markers, filler words. They stay in the full normalized name.
NAME_CORE_DROP = {
    # legal forms
    "private", "limited", "llp", "llc", "lp", "pllc", "pc", "plc",
    "incorporated", "corporation", "company", "opc", "gmbh", "ag", "bv", "nv",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "scm",
    "etablissements",
    # honorifics / markers
    "sri", "smt", "mr", "mrs", "ms", "dr", "messrs", "dba", "aka", "fka",
    # filler
    "the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d",
    "en", "au", "aux", "a", "an",
}

# --------------------------------------------------------------------------
# Addresses

US_ADDRESS_TOKEN_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bld": "boulevard", "boul": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "cir": "circle", "hwy": "highway", "hiway": "highway",
    "pkwy": "parkway", "pky": "parkway", "pl": "place", "sq": "square",
    "ter": "terrace", "terr": "terrace", "trl": "trail", "cres": "crescent",
    "pt": "point", "mt": "mount", "ft": "fort", "rte": "route", "rt": "route",
    "twp": "township", "expy": "expressway", "fwy": "freeway", "tpke": "turnpike",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "rm": "room", "dept": "department",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

INDIA_ADDRESS_TOKEN_MAP = {
    "rd": "road", "st": "street", "marg": "road", "nr": "near", "opp": "opposite",
    "bldg": "building", "flr": "floor", "fl": "floor", "hno": "house", "h": "house",
    "dist": "district", "distt": "district", "tal": "taluk", "tq": "taluk", "tk": "taluk",
    "ngr": "nagar", "mkt": "market", "sec": "sector", "ph": "phase",
    "extn": "extension", "ext": "extension", "clny": "colony", "col": "colony",
    "apt": "apartment", "apts": "apartments", "soc": "society", "chs": "society",
    "vill": "village", "vil": "village", "po": "post", "ps": "police station",
    "ind": "industrial", "indl": "industrial", "estt": "estate",
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "poona": "pune", "gurgaon": "gurugram",
    "pondicherry": "puducherry", "trivandrum": "thiruvananthapuram",
    "baroda": "vadodara", "mysore": "mysuru", "calicut": "kozhikode",
    "cochin": "kochi", "vizag": "visakhapatnam", "banaras": "varanasi",
    "benares": "varanasi", "orissa": "odisha", "cawnpore": "kanpur",
    "belgaum": "belagavi", "mangalore": "mangaluru", "hubli": "hubballi",
}

FRANCE_ADDRESS_TOKEN_MAP = {
    "r": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "bvd": "boulevard", "av": "avenue", "ave": "avenue", "ch": "chemin",
    "chem": "chemin", "imp": "impasse", "pl": "place", "rte": "route",
    "all": "allee", "al": "allee", "sq": "square", "fg": "faubourg",
    "fbg": "faubourg", "pas": "passage", "pass": "passage", "qu": "quai",
    "qua": "quai", "crs": "cours", "cr": "cours", "res": "residence",
    "lot": "lotissement", "st": "saint", "ste": "sainte", "apt": "appartement",
    "app": "appartement", "appt": "appartement", "bat": "batiment",
    "bt": "batiment", "esc": "escalier", "etg": "etage", "zi": "zone industrielle",
    "za": "zone artisanale", "prom": "promenade", "mte": "montee",
}

GENERIC_ADDRESS_TOKEN_MAP = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "apt": "apartment", "bldg": "building", "fl": "floor",
}

# Tokens dropped from addresses everywhere: numbering words and noise markers.
# ("co" is deliberately absent: it is Colorado's state code. "c/o" is removed
# by a regex in normalize_master, so single letters like the "c" in "1-C" survive.)
ADDRESS_DROP_TOKENS = {"no", "number", "nos", "num", "null", "none", "nil"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "washington dc": "dc", "puerto rico": "pr",
    "guam": "gu", "virgin islands": "vi",
}

INDIA_STATE_CODES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa",
    "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "or": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana",
    "tg": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
    "an": "andaman and nicobar islands", "ch": "chandigarh",
    "dn": "dadra and nagar haveli", "dd": "daman and diu", "dl": "delhi",
    "jk": "jammu and kashmir", "la": "ladakh", "ld": "lakshadweep",
    "py": "puducherry", "nct of delhi": "delhi", "new delhi": "delhi",
    "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
}

# Whole-component maps, keyed by lowercase country label. Unknown countries
# (open set) simply get no component map.
# Single-token components that are state codes are protected from the
# address token map (US "CT"/"FL"/"NE"/"MT" would otherwise become
# court/floor/northeast/mount).
STATE_CODES = {
    "us": set(US_STATES.values()),
    "india": {k for k in INDIA_STATE_CODES if " " not in k},
}

COMPONENT_MAPS = {
    "us": US_STATES,
    "india": INDIA_STATE_CODES,
}

ADDRESS_TOKEN_MAPS = {
    "us": US_ADDRESS_TOKEN_MAP,
    "india": INDIA_ADDRESS_TOKEN_MAP,
    "france": FRANCE_ADDRESS_TOKEN_MAP,
}

# "&" is "and" in English, "et" in French.
AMPERSAND = {"france": " et "}
