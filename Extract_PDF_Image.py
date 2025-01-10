import os
import logging
from pathlib import Path
import fitz  # PyMuPDF
import tempfile
import base64

# 1) OpenAI �Ȥ�� (�I�s gpt-4o-mini)
from openai import OpenAI

# 2) LangChain / Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# ---------------------------
# �]�w API keys
# ---------------------------
os.environ['OPENAI_API_KEY'] = 'sk-xQfaQBL6rNJ8Ktg6V_GEWqjZK93fTsPM63_qhqPq80T3BlbkFJLEHuKq1TIIbyEdT51UZomtm00HwQwziNJJTgVRmaMA'
os.environ['PINECONE_API_KEY'] = 'pcsk_6u2B7E_HSPxt21tmWCpyCSQMpEx1Y8PoeoRC9hSmwPCfuJnkH1auTgTabD2aoFBogiZTcc'

index_name = "leohw2"

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Directory to save extracted images
# ---------------------------
IMAGE_DIRECTORY = Path(r"leohw2")
IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)

# ---------------------------
# PDF Read/Extract functions
# ---------------------------
def download_pdf(pdf_url):
    """
    Reads a local PDF file, writes it to a temporary file, 
    and returns both the temporary file path and the PDF bytes.
    """
    logger.info("Reading local PDF from path...")
    with open(pdf_url, 'rb') as f:
        pdf_bytes = f.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        tmp_pdf_path = tmp_pdf.name

    logger.info(f"PDF read from local path and saved to temporary file: {tmp_pdf_path}")
    return tmp_pdf_path, pdf_bytes

def extract_images_from_pdf_bytes(pdf_bytes, output_dir):
    """
    Extracts images from the given PDF bytes using PyMuPDF, 
    saves them to the specified output directory, 
    and returns a list of the saved image file paths.
    """
    logger.info("Extracting images from PDF...")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        logger.debug(f"Found {len(image_list)} images on page {page_num + 1}.")

        xref_seen = set()
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            if xref in xref_seen:
                continue  # Skip duplicate images
            xref_seen.add(xref)

            pix = fitz.Pixmap(pdf_document, xref)

            # Handle transparency
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Convert to RGB if necessary
            if pix.colorspace.n > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            image_filename = output_dir / f"page_{page_num + 1}_image_{img_index + 1}.png"
            pix.save(str(image_filename))
            image_paths.append(str(image_filename))
            pix = None  # Free memory
            logger.debug(f"Saved image: {image_filename}")

    pdf_document.close()
    logger.info(f"Extracted and saved {len(image_paths)} images.")
    return image_paths

def encode_image(image_path):
    """
    Encodes the specified image file in Base64 and returns the encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # 1) �إ� OpenAI �Τ��
    client = OpenAI()

    # 2) ��l�� Pinecone Vector Store
    embeddings = OpenAIEmbeddings() 
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # 3) ���w�z�����a PDF ���|
    pdf_url = r"D:\Extract_PDF_Image-master\Paper2.pdf"

    try:
        # (A) Ū�� PDF / �ন�Ȧs��
        tmp_pdf_path, pdf_bytes = download_pdf(pdf_url)

        # (B) �q PDF ���^���Ϥ� (list of image_paths)
        image_paths = extract_images_from_pdf_bytes(pdf_bytes, IMAGE_DIRECTORY)
        logger.info(f"Images successfully extracted and saved: {image_paths}")

        # (C) �N�C�i�Ϥ��e�� GPT �ҫ��A�ç�^���g�i Pinecone
        for extracted_image_path in image_paths:
            # ��Ϥ��s�X�� Base64�]�Y�Ϥ��D�`�j�A�i��|�z token�F�i�ۦ�I�_�����Y�^
            base64_image = encode_image(extracted_image_path)

            # �I�s gpt-4o-mini �ҫ�
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"What is in this image: {extracted_image_path}? Please respond **only in English**.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )

            # ���X GPT �^���¤�r
            gpt_response_text = response.choices[0].message.content
            print("Type of gpt_response_text =", type(gpt_response_text))
            # ���o�ɦW (e.g. "page_1_image_1.png")
            filename_only = os.path.basename(extracted_image_path)

            logger.info(f"GPT response for {filename_only}:\n{gpt_response_text}\n")

            # (D) �g�i Pinecone
            # - texts=[gpt_response_text] �|�Q�Ψӭp��V�q
            # - metadata�G�ۭq���
            vectorstore = PineconeVectorStore.from_texts(
                texts=[gpt_response_text],
                metadatas=[{"source": filename_only, "text": gpt_response_text}],
                embedding=embeddings,
                index_name=index_name
            )

    finally:
        # �����Ȧs PDF ��
        if os.path.exists(tmp_pdf_path):
            ##os.remove(tmp_pdf_path)
            logger.info(f"Temporary PDF file {tmp_pdf_path} removed.")
