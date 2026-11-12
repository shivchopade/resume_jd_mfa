"""
make_sample_pdfs.py - generates the six demo resumes in sample_data/resumes/.

All people are fictional. Each resume tests one part of the system:

    ananya_sharma.pdf   strong fit (HR label 3). Has a dark header band with WHITE text,
                        which is visible and must NOT be flagged as hidden text.
    rahul_verma.pdf     good fit (2): Python, SQL, Flask, Git but no Docker / AWS.
    sneha_patil.pdf     weak fit (1): data analyst - Python + SQL, no backend work.
    priya_joshi.pdf     weak fit (1) that copies sentences from the JD  -> flag jd_copy
    amit_kulkarni.pdf   not relevant (0): frontend developer.
    rohan_deshmukh.pdf  not relevant (0) that hides keywords in white text and 1 pt text
                        -> flags hidden_text + tiny_font

Run:  python scripts/make_sample_pdfs.py
"""

import textwrap
from pathlib import Path

import pymupdf

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "resumes"
PAGE_W, PAGE_H, MARGIN = 595, 842, 56  # A4 in points


class ResumeWriter:
    """Tiny helper that writes text top-to-bottom on an A4 page."""

    def __init__(self):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = 64

    def line(self, text, size=10.5, bold=False, color=(0, 0, 0), x=MARGIN):
        font = "hebo" if bold else "helv"  # Helvetica Bold / Helvetica
        self.page.insert_text((x, self.y), text, fontsize=size, fontname=font, color=color)
        self.y += size * 1.45

    def para(self, text, size=10.5, width=92):
        for piece in textwrap.wrap(text, width):
            self.line(piece, size)

    def bullet(self, text, width=88):
        pieces = textwrap.wrap(text, width)
        for i, piece in enumerate(pieces):
            self.line(("• " if i == 0 else "   ") + piece)

    def heading(self, text):
        self.y += 9
        self.line(text.upper(), size=11.5, bold=True)
        self.y += 1

    def header(self, name, contact, dark=False):
        if dark:  # dark band with white text: VISIBLE, must not count as hidden
            self.page.draw_rect(pymupdf.Rect(0, 0, PAGE_W, 96), color=None, fill=(0.08, 0.08, 0.08))
            self.y = 46
            self.line(name, size=20, bold=True, color=(1, 1, 1))
            self.line(contact, size=10, color=(1, 1, 1))
            self.y = 128
        else:
            self.line(name, size=20, bold=True)
            self.line(contact, size=10, color=(0.3, 0.3, 0.3))
            self.y += 4

    def save(self, filename):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.doc.save(OUT_DIR / filename, garbage=3, deflate=True)
        self.doc.close()


def ananya():
    w = ResumeWriter()
    w.header("Ananya Sharma", "Pune | ananya.sharma@example.com | github.com/ananya-dev", dark=True)
    w.heading("Summary")
    w.para("Backend developer focused on Python services. Built and deployed REST APIs used by "
           "real users, with clean code, tests and containerised deployments.")
    w.heading("Skills")
    w.para("Python, FastAPI, Flask, SQL, PostgreSQL, Redis, Docker, Docker Compose, AWS (EC2, S3), "
           "Git, GitHub Actions, pytest, Linux")
    w.heading("Experience")
    w.line("Backend Developer Intern - CloudNest Labs, Pune (Jan 2026 - Jun 2026)", bold=True)
    w.bullet("Built 14 REST API endpoints in FastAPI for an internal hiring dashboard.")
    w.bullet("Designed PostgreSQL tables and optimised slow SQL queries from 2.1 s to 180 ms.")
    w.bullet("Containerised three services with Docker and deployed them on AWS EC2.")
    w.bullet("Reviewed pull requests on GitHub and added pytest unit tests (85% coverage).")
    w.heading("Projects")
    w.bullet("Job Board API - FastAPI + PostgreSQL + Docker, JWT auth, deployed on AWS.")
    w.bullet("URL Shortener - Flask + Redis cache, CI pipeline with GitHub Actions.")
    w.heading("Education")
    w.para("B.E. Information Technology, Savitribai Phule Pune University - 2026, CGPA 8.7")
    w.save("ananya_sharma.pdf")


def rahul():
    w = ResumeWriter()
    w.header("Rahul Verma", "Nashik | rahul.verma@example.com")
    w.heading("Skills")
    w.para("Python, Flask, SQL, MySQL, HTML, Git, REST APIs")
    w.heading("Experience")
    w.line("Software Developer Intern - Shree Infotech, Nashik (Jul 2025 - Dec 2025)", bold=True)
    w.bullet("Built REST APIs with Flask for a college admission portal.")
    w.bullet("Wrote SQL queries and stored procedures on MySQL for student records.")
    w.bullet("Used Git and GitHub for version control in a team of four developers.")
    w.heading("Projects")
    w.bullet("Library Management API - Flask + MySQL with login and book issue tracking.")
    w.bullet("Expense Tracker - Python CLI app that stores data in SQLite.")
    w.heading("Education")
    w.para("B.Tech Computer Engineering, Dr. Babasaheb Ambedkar Technological University - 2025")
    w.save("rahul_verma.pdf")


def sneha():
    w = ResumeWriter()
    w.header("Sneha Patil", "Kolhapur | sneha.patil@example.com")
    w.heading("Summary")
    w.para("Data analyst who turns raw business data into clear dashboards and reports.")
    w.heading("Skills")
    w.para("Python (pandas, matplotlib), SQL, Excel, Power BI, Tableau, statistics")
    w.heading("Experience")
    w.line("Data Analyst Intern - RetailMart, Kolhapur (Feb 2026 - May 2026)", bold=True)
    w.bullet("Wrote SQL queries to pull weekly sales data and cleaned it with pandas.")
    w.bullet("Built Power BI dashboards that track inventory for 40 stores.")
    w.heading("Projects")
    w.bullet("Customer churn analysis - Python and Excel, logistic regression baseline.")
    w.heading("Education")
    w.para("B.Sc. Statistics, Shivaji University - 2025")
    w.save("sneha_patil.pdf")


def priya():
    w = ResumeWriter()
    w.header("Priya Joshi", "Aurangabad | priya.joshi@example.com")
    w.heading("Skills")
    w.para("Python, C, HTML, MS Office")
    w.heading("Experience")
    w.line("Trainee - Akshar Software Services (Mar 2026 - May 2026)", bold=True)
    # These two lines are copied word-for-word from the JD -> should be flagged "jd_copy"
    w.bullet("Design, build and maintain scalable REST APIs using Python and FastAPI or Flask.")
    w.bullet("Write efficient SQL queries and manage relational databases such as PostgreSQL or MySQL.")
    w.heading("Projects")
    w.bullet("Student result calculator in Python.")
    w.bullet("Personal portfolio website using HTML.")
    w.heading("Education")
    w.para("B.E. Computer Engineering, Dr. Babasaheb Ambedkar Marathwada University - 2026")
    w.save("priya_joshi.pdf")


def amit():
    w = ResumeWriter()
    w.header("Amit Kulkarni", "Mumbai | amit.kulkarni@example.com")
    w.heading("Summary")
    w.para("Frontend developer who builds fast, accessible user interfaces.")
    w.heading("Skills")
    w.para("JavaScript, TypeScript, React, Next.js, HTML, CSS, Tailwind, Figma, Git")
    w.heading("Experience")
    w.line("Frontend Developer Intern - PixelCraft Studio, Mumbai (Aug 2025 - Jan 2026)", bold=True)
    w.bullet("Built responsive landing pages in React and Next.js from Figma designs.")
    w.bullet("Improved Lighthouse performance score from 62 to 95.")
    w.heading("Projects")
    w.bullet("Movie search app with React and a public REST API.")
    w.heading("Education")
    w.para("B.E. Electronics and Telecommunication, University of Mumbai - 2025")
    w.save("amit_kulkarni.pdf")


def rohan():
    w = ResumeWriter()
    w.header("Rohan Deshmukh", "Satara | rohan.deshmukh@example.com")
    w.heading("Summary")
    w.para("Customer support executive with strong communication skills and a calm attitude.")
    w.heading("Skills")
    w.para("Communication, MS Office, CRM tools, customer handling, Hindi, Marathi, English")
    w.heading("Experience")
    w.line("Customer Support Executive - QuickServe BPO, Pune (2024 - 2026)", bold=True)
    w.bullet("Handled 60+ customer calls per day with a 92% satisfaction score.")
    w.bullet("Logged and tracked customer complaints in the CRM system.")
    w.heading("Education")
    w.para("B.Com, Shivaji University - 2024")
    # Resume gaming 1: keywords in WHITE text on a white page (invisible to humans)
    w.page.insert_text((MARGIN, 770), "Python SQL Docker AWS Git FastAPI PostgreSQL Kubernetes REST API",
                       fontsize=10, color=(1, 1, 1))
    # Resume gaming 2: keywords in 1 pt text (too small to read)
    w.page.insert_text((MARGIN, 790), "Python backend developer with Docker SQL AWS experience",
                       fontsize=1)
    w.save("rohan_deshmukh.pdf")


if __name__ == "__main__":
    for build in (ananya, rahul, sneha, priya, amit, rohan):
        build()
    print(f"Wrote 6 sample resumes to {OUT_DIR}")
