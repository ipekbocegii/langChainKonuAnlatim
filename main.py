from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# .env dosyasındaki çevresel değişkenleri yükle
load_dotenv()

# ChatOpenAI modelini seçiyoruz
llm = ChatOpenAI(model="gpt-4")

# Prompt template (şablon) tanımlanıyor
template = "Bir {tone} e-postası yaz, {company} şirketine {position} pozisyonuna olan ilgimi belirten ve {skill} becerimi ana güçlü yön olarak vurgulayan. Mesajı 4 satırla sınırlı tut."
# ChatPromptTemplate ile şablonu oluşturuyoruz
prompt_template = ChatPromptTemplate.from_template(template)
print(prompt_template)

prompt = prompt_template.invoke({
    "tone": "energetic",
    "company":"Softtech",
    "position":"AI Engineer",
    "skill":"AI"
}
    
)
# Promptu yazdırıyoruz
print(prompt)
result = llm.invoke(prompt)
print(result.content)