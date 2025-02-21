from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Farklı geri bildirim türleri için prompt şablonlarını tanımla
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Sen yardımcı bir asistansın."),
        ("human",
         "Bu pozitif geri bildirim için bir teşekkür mesajı oluştur:  {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Sen yardımcı bir asistansın."),
        ("human",
         "Bu olumsuz geri bildirime cevap oluştur: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Sen yardımcı bir asistansın."),
        (
            "human",
            "Bu nötr geri bildirim için daha fazla bilgi isteği oluştur: {feedback}.",
        ),
    ]
)



escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Sen yardımcı bir asistansın"),
        (
            "human",
            "Bu geri bildirimi insan temsilcisine iletmek için bir mesaj oluştur: {feedback}.",
        ),
    ]
)

# Geri bildirim sınıflandırma şablonunu tanımla
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Sen yardımcı bir asistansın."),
        ("human",
         "Bu geri bildirimi pozitif, negatif, nötr ya da insan temsilcisine iletme olarak sınıflandır: {feedback}."),
    ]
)

# Geri bildirimleri işlemek için runnable dallarını (branches) tanımla
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Sınıflandırma zincirini oluştur
classification_chain = classification_template | model | StrOutputParser()

# Sınıflandırma ve geri bildirim oluşturmayı birleştir
chain = classification_chain | branches

# Zinciri bir örnek geri bildirimle çalıştır
# İyi geri bildirim - "Ürün mükemmel. Gerçekten kullanmayı çok sevdim ve çok faydalı buldum."
# Kötü geri bildirim - "Ürün berbat. Sadece bir kullanımdan sonra bozuldu ve kalite çok kötü."
# Nötr geri bildirim - "Ürün iyi. Beklediğim gibi çalışıyor ama olağanüstü bir şey değil."
# Varsayılan - "Ürün hakkında henüz bir fikrim yok. Özellikleri ve faydaları hakkında daha fazla bilgi verebilir misiniz?"


review = "Ürün berbat. Sadece bir kullanımdan sonra bozuldu ve kalite çok kötü."
result = chain.invoke({"feedback": review})

# Output the result
print(result)