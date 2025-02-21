from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ChatOpenAI modelini seçiyoruz
llm = ChatOpenAI(model="gpt-4")

# Mesajları tanımla (System ve Human mesajları)
messages = [
    ("system", "Sen, {konu} hakkında şakalar yapan bir komedyensin."),
    ("human", "Bana {saka_sayisi} tane şaka yapar mısın?")
]

# ChatPromptTemplate ile mesajları kullanarak bir şablon oluştur
prompt_template = ChatPromptTemplate.from_messages(messages)

# Şablonla birlikte parametreleri geçerek mesajı oluştur
prompt = prompt_template.invoke({
    "konu": "yazılımcı",  # Konu: yazılımcı
    "saka_sayisi": 3  # 3 tane şaka
})

# Sonucu yazdır
print("\n-------- system ve human Mesajları ile Prompt --------\n")
print(prompt)

result = llm.invoke(prompt)
print(result.content)