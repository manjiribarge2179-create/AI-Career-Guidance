"""
SQLite database for Career Guidance Web App
Stores careers, streams, colleges, and exam information
"""

import sqlite3
import os

DB_PATH = "data/career_guidance.db"


def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Streams table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS streams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class_level TEXT NOT NULL,  -- '10th' or '12th'
            description TEXT,
            subjects TEXT
        )
    """)

    # Careers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS careers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            stream_id INTEGER,
            description TEXT,
            required_exams TEXT,
            avg_salary TEXT,
            duration TEXT,
            FOREIGN KEY (stream_id) REFERENCES streams(id)
        )
    """)

    # Entrance Exams table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entrance_exams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            full_name TEXT,
            stream TEXT,
            class_level TEXT,
            description TEXT,
            conducting_body TEXT
        )
    """)

    # Colleges table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS colleges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            stream TEXT,
            location TEXT,
            ranking TEXT,
            website TEXT
        )
    """)

    # Chat history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    _seed_data(cur, conn)
    conn.close()
    print("[DB] Database initialized successfully.")


def _seed_data(cur, conn):
    # Only seed if empty
    cur.execute("SELECT COUNT(*) FROM streams")
    if cur.fetchone()[0] > 0:
        return

    streams = [
        ("Science PCM", "12th", "Physics, Chemistry, Math — for engineering and tech careers",
         "Physics, Chemistry, Mathematics"),
        ("Science PCB", "12th", "Physics, Chemistry, Biology — for medical and life science careers",
         "Physics, Chemistry, Biology"),
        ("Science PCMB", "12th", "All four science subjects — keeps all options open",
         "Physics, Chemistry, Mathematics, Biology"),
        ("Commerce", "12th", "Business, finance, and accounting focused stream",
         "Accountancy, Business Studies, Economics, Mathematics (optional)"),
        ("Arts/Humanities", "12th", "Social sciences, languages, and creative fields",
         "History, Political Science, Psychology, Sociology, Geography"),
        ("Polytechnic Diploma", "10th", "3-year technical diploma in engineering trades",
         "Technical subjects based on trade"),
        ("ITI", "10th", "1-2 year vocational trades for direct employment",
         "Trade-specific practical and theory subjects"),
    ]
    cur.executemany(
        "INSERT INTO streams (name, class_level, description, subjects) VALUES (?,?,?,?)",
        streams
    )

    careers = [
        ("Software Engineer", 1, "Develops software applications and systems",
         "JEE, CUET", "4-15 LPA", "B.Tech 4 years"),
        ("Doctor (MBBS)", 2, "Medical professional treating patients",
         "NEET", "8-30 LPA", "MBBS 5.5 years"),
        ("Chartered Accountant", 4, "Manages accounts, audits, and taxation",
         "CA Foundation/Intermediate/Final", "6-20 LPA", "4-5 years"),
        ("Civil Services (IAS/IPS)", 3, "Government administrative and police services",
         "UPSC", "Varies", "After graduation"),
        ("Architect", 1, "Designs buildings and structures",
         "NATA, JEE Paper 2", "4-12 LPA", "B.Arch 5 years"),
        ("Lawyer", 5, "Legal professional practicing in courts",
         "CLAT", "4-15 LPA", "BA LLB 5 years"),
        ("Fashion Designer", 5, "Designs clothing and accessories",
         "NIFT, NID", "3-10 LPA", "B.Des 4 years"),
        ("Data Scientist", 1, "Analyzes large data to extract insights",
         "JEE", "6-20 LPA", "B.Tech/B.Sc 4 years"),
        ("Journalist", 5, "Reports news across print, digital, TV",
         "IIMC entrance, Symbiosis", "3-10 LPA", "BA/BMM 3 years"),
        ("Pharmacist", 2, "Dispenses medicines and advises patients",
         "GPAT, State Pharmacy exams", "3-8 LPA", "B.Pharma 4 years"),
        ("Pilot (Commercial)", 1, "Flies commercial aircraft",
         "DGCA CPL", "8-30 LPA", "CPL 2-3 years"),
        ("Hotel Manager", 5, "Manages hotel operations and hospitality",
         "NCHMCT JEE", "4-12 LPA", "BHM 4 years"),
        ("Psychologist", 5, "Studies and treats mental health",
         "CUET, university entrance", "4-15 LPA", "BSc/BA + MSc 5 years"),
        ("Veterinary Doctor", 2, "Treats animals and livestock",
         "State Vet CET", "4-12 LPA", "BVSc 5.5 years"),
        ("Army Officer", 1, "Commissioned officer in Indian Army",
         "NDA, CDS", "Varies (govt pay)", "NDA 3 years + IMA 1 year"),
    ]
    cur.executemany(
        "INSERT INTO careers (title, stream_id, description, required_exams, avg_salary, duration) VALUES (?,?,?,?,?,?)",
        careers
    )

    exams = [
        ("JEE Main", "Joint Entrance Examination Main", "PCM", "12th",
         "National entrance for NITs, IIITs and other engineering colleges", "NTA"),
        ("JEE Advanced", "Joint Entrance Examination Advanced", "PCM", "12th",
         "Entrance exam for IITs", "IIT"),
        ("NEET", "National Eligibility cum Entrance Test", "PCB", "12th",
         "Medical entrance for MBBS, BDS, BAMS and other medical courses", "NTA"),
        ("CLAT", "Common Law Admission Test", "Any", "12th",
         "Entrance for National Law Universities", "CLAT Consortium"),
        ("NIFT", "National Institute of Fashion Technology Entrance", "Any", "12th",
         "Entrance for fashion design colleges", "NIFT"),
        ("NID DAT", "National Institute of Design Design Aptitude Test", "Any", "12th",
         "Entrance for product and communication design", "NID"),
        ("NATA", "National Aptitude Test in Architecture", "PCM", "12th",
         "Entrance for architecture programmes", "CoA"),
        ("NDA", "National Defence Academy", "PCM", "12th",
         "Entrance for Army, Navy and Air Force after 12th", "UPSC"),
        ("CUET", "Common University Entrance Test", "Any", "12th",
         "Entrance for central universities undergraduate admissions", "NTA"),
        ("NCHMCT JEE", "National Council for Hotel Management Joint Entrance Exam", "Any", "12th",
         "Entrance for hotel management institutes", "NTA"),
        ("CA Foundation", "Chartered Accountancy Foundation", "Commerce", "12th",
         "First step for Chartered Accountancy course", "ICAI"),
        ("IPMAT", "Integrated Program in Management Aptitude Test", "Any", "12th",
         "5-year BBA+MBA programme at IIM Indore and Rohtak", "IIMs"),
    ]
    cur.executemany(
        "INSERT INTO entrance_exams (name, full_name, stream, class_level, description, conducting_body) VALUES (?,?,?,?,?,?)",
        exams
    )

    colleges = [
        ("IIT Bombay", "Government", "Engineering", "Mumbai", "Top 3 in India", "www.iitb.ac.in"),
        ("IIT Delhi", "Government", "Engineering", "New Delhi", "Top 3 in India", "www.iitd.ac.in"),
        ("AIIMS Delhi", "Government", "Medical", "New Delhi", "Top Medical College", "www.aiims.edu"),
        ("NLU Delhi", "Government", "Law", "New Delhi", "Top Law School", "www.nludelhi.ac.in"),
        ("NIFT Delhi", "Government", "Fashion Design", "New Delhi", "Top Fashion Institute", "www.nift.ac.in"),
        ("NIT Trichy", "Government", "Engineering", "Tiruchirappalli", "Top NIT", "www.nitt.edu"),
        ("BITS Pilani", "Private", "Engineering", "Pilani", "Top Private Engineering", "www.bits-pilani.ac.in"),
        ("Christ University", "Private", "Commerce/BBA", "Bangalore", "Top BBA College", "www.christuniversity.in"),
        ("Symbiosis Pune", "Private", "Management/Media", "Pune", "Top Private University", "www.siu.edu.in"),
        ("IHM Mumbai", "Government", "Hotel Management", "Mumbai", "Top Hotel Management", "www.ihmctan.edu"),
    ]
    cur.executemany(
        "INSERT INTO colleges (name, type, stream, location, ranking, website) VALUES (?,?,?,?,?,?)",
        colleges
    )

    conn.commit()
    print("[DB] Seed data inserted.")


def save_chat(session_id, user_msg, bot_response):
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_history (session_id, user_message, bot_response) VALUES (?,?,?)",
        (session_id, user_msg, bot_response)
    )
    conn.commit()
    conn.close()


def get_careers_by_stream(stream_name):
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.title, c.description, c.required_exams, c.avg_salary, c.duration
        FROM careers c
        JOIN streams s ON c.stream_id = s.id
        WHERE s.name LIKE ?
    """, (f"%{stream_name}%",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_exams():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM entrance_exams").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_colleges_by_stream(stream):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM colleges WHERE stream LIKE ?", (f"%{stream}%",)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
