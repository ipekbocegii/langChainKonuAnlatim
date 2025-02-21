import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


# Metin dosyasının bulunduğu dizini ve kalıcı dizini tanımla
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Chroma vektör mağazasının zaten var olup olmadığını kontrol et
if not os.path.exists(persistent_directory):
    print("Kalıcı dizin mevcut değil. Vektör mağazası başlatılıyor...")

    # Metin dosyasının mevcut olup olmadığını kontrol et
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} dosyası mevcut değil. Lütfen yolu kontrol edin."
        )

    # Metin içeriğini dosyadan oku
    loader = TextLoader(file_path)
    documents = loader.load()

    # dokumanı chunklara ayır
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50) 
    docs = text_splitter.split_documents(documents)

    # Bölünmüş belgeler hakkında bilgi göster
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Gerekirse geçerli bir embedding modeline güncelleme yapın
    print("\n--- Finished creating embeddings ---")

    # Vektör store oluştur ve otomatik olarak kalıcı hale getir
    print("\n--- Vektör store oluşturuluyor ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- --- Vektör mağazası oluşturulması tamamlandı --- ---")

else:
    print("Vektör mağazası zaten mevcut. Başlatmaya gerek yok.")




# Sorular
# Yüzük Taşıyıcısı kimdir?
# Gandalf, Frodo ile nerede buluşur?