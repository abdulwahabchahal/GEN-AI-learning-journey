import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini = genai.Client()
EMBED_MODEL = "gemini-embedding-001"

records = [
    {"id": "faq-01", "title": "Application Deadline", "category": "Admissions",
     "text": "Applications for the fall semester must be submitted by June 30."},
    {"id": "faq-02", "title": "Tuition Fees", "category": "Fees",
     "text": "Undergraduate tuition is $4,500 per semester, payable in two installments."},
    {"id": "faq-03", "title": "Scholarships", "category": "Fees",
     "text": "Merit scholarships cover up to 75% of tuition for students with a GPA above 3.8."},
    {"id": "faq-04", "title": "Hostel Accommodation", "category": "Campus Life",
     "text": "On-campus hostels offer shared and single rooms, with meals included in the monthly rent."},
    {"id": "faq-05", "title": "Library Hours", "category": "Campus Life",
     "text": "The central library is open from 8 AM to midnight during exams and 8 AM to 8 PM otherwise."},
    {"id": "faq-06", "title": "Course Registration", "category": "Academics",
     "text": "Students register for courses through the online student portal during the first two weeks of each semester."},
    {"id": "faq-07", "title": "Grading Policy", "category": "Academics",
     "text": "A student must score at least 50% in every course to pass, and a GPA below 2.0 leads to academic probation."},
    {"id": "faq-08", "title": "Internships", "category": "Careers",
     "text": "The career office arranges summer internships with partner companies for third-year students."},
    {"id": "faq-09", "title": "Transport Service", "category": "Campus Life",
     "text": "University buses run from five city locations to the campus every 30 minutes on weekdays."},
    {"id": "faq-10", "title": "Student Counseling", "category": "Student Support",
     "text": "Free confidential counseling sessions are available for all students by appointment."},
]

def embed_documents(texts):
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in result.embeddings]

# Create a local, persistent Chroma database (saved in the chroma_db folder)
chroma = chromadb.PersistentClient(path="chroma_db")

# Start fresh so re-running the script doesn't create duplicates
try:
    chroma.delete_collection("university_faqs")
except Exception:
    pass

collection = chroma.create_collection(
    name="university_faqs",
    metadata={"hnsw:space": "cosine"},   # use cosine distance
)

vectors = embed_documents([r["text"] for r in records])

collection.add(
    ids=[r["id"] for r in records],
    documents=[r["text"] for r in records],
    embeddings=vectors,
    metadatas=[{"title": r["title"], "category": r["category"], "source": "University FAQ page"}
               for r in records],
)

# Show what is stored (good screenshot material)
print("Records stored:", collection.count())
sample = collection.get(ids=["faq-02"], include=["documents", "metadatas", "embeddings"])
print("\nSample record")
print("  ID:       ", sample["ids"][0])
print("  Document: ", sample["documents"][0])
print("  Metadata: ", sample["metadatas"][0])
print("  Vector length:", len(sample["embeddings"][0]))
print("  First 5 values:", [round(float(x), 4) for x in sample["embeddings"][0][:5]])