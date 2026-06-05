# universe.py now just provides a helper to dynamically get the universe from the DB.
from fetch_eod import get_universe_from_db, get_db_connection

def get_current_universe():
    conn = get_db_connection()
    u = get_universe_from_db(conn)
    conn.close()
    return u
