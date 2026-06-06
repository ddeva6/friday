from datetime import datetime, timedelta
import sqlite3

def get_holidays(db_path="test.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT date FROM holidays")
    holidays = {row[0] for row in c.fetchall()}
    conn.close()
    return holidays

def is_trading_day(dt, holidays):
    if dt.weekday() >= 5: # 5 = Saturday, 6 = Sunday
        return False
    if dt.strftime('%Y-%m-%d') in holidays:
        return False
    return True

def next_session(date_str, db_path="test.db"):
    holidays = get_holidays(db_path)
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    dt += timedelta(days=1)
    while not is_trading_day(dt, holidays):
        dt += timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

def add_sessions(date_str, n, db_path="test.db"):
    holidays = get_holidays(db_path)
    dt = datetime.strptime(date_str, '%Y-%m-%d')

    sessions_added = 0
    while sessions_added < n:
        dt += timedelta(days=1)
        if is_trading_day(dt, holidays):
            sessions_added += 1

    return dt.strftime('%Y-%m-%d')
