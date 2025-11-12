from langchain_community.document_loaders import PyPDFLoader

# Load the college website landing page PDF
pdf_path = "Aurora_University_LandingPage.pdf"   # file should be in same folder
loader = PyPDFLoader(pdf_path)

# Load all pages
pages = loader.load()

# Print confirmation and a sample of the text
print(f"✅ Loaded {len(pages)} pages from the college landing page.")
print("\n--- First few lines from your PDF ---\n")
print(pages[0].page_content[:500])

# python upload_pdf.py (for running the pdf)