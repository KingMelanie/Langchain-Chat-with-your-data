import os
from openai import OpenAI
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

client = OpenAI(api_key  = os.getenv('OPENAI_API_KEY'))


# ## PDFs
# 
# Let's load a PDF [transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf) from Andrew Ng's famous CS229 course! These documents are the result of automated transcription so words and sentences are sometimes split unexpectedly.

# In[ ]:


# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
#! pip install pypdf 


# In[ ]:


from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://www.sfbu.edu/sites/default/files/2022-12/2023Catalog.pdf")
pages = loader.load()


# Each page is a `Document`.
# 
# A `Document` contains text (`page_content`) and `metadata`.

# In[ ]:


len(pages)


# In[ ]:


page = pages[0]


# In[ ]:


print(page.page_content[0:500])


# In[ ]:


page.metadata


# ## YouTube

# In[ ]:


from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


# In[ ]:


# ! pip install yt_dlp
# ! pip install pydub


# **Note**: This can take several minutes to complete.

# In[ ]:


url="https://youtu.be/AuDodQm7nm8?si=QgtvcsNofH8vqbn0"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()


# In[ ]:


docs[0].page_content[0:500]


# ## URLs

# In[ ]:


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://mailchimp.com/")


# In[ ]:


docs = loader.load()


# In[ ]:


print(docs[0].page_content[:500])



