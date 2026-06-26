import base64
from io import BytesIO
from PIL import Image

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class ImageSummarizer:
    def __init__(self):
        # Ensure you use a valid Groq vision model for summarization
        self.model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    
    def compress_image(self, image_bytes, max_size=(512, 512), quality=60):
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB to prevent issues with RGBA/PNG saving as JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image.thumbnail(max_size)
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)

        return buffer.getvalue()

    def summarize_image(self, image_bytes):
        try:
            compressed_bytes = self.compress_image(image_bytes)
            image_b64 = base64.b64encode(compressed_bytes).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """
Describe this document image in detail for a search retrieval index.

Focus strictly on extracting meaning from:
- diagrams and architectures
- charts and data trends
- tables and their values
- text labels

Keep the description concise but highly descriptive so it can be searched.
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ]
            )

            response = self.model.invoke([message])
            return response.content
        except Exception as e:
            print(f"Failed to summarize image: {e}")
            return "Image could not be summarized."