from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
#StrOutputParser genellikle daha önce tanımlanan çıktıları analiz etmek ve sonucu kullanmaya uygun hale getirmek için kullanılır
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4")

# Film özeti için prompt şablonunu tanımla
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Bir film eleştirmenisin."),
        ("human", "Lütfen {movie_name} filminin kısa bir özetini ver."),
    ]
)

# Konu analiz adımını tanımla
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Bir film eleştirmenisin."),
            ("human", "Şu konuya göz at: {plot}. Güçlü ve zayıf yönleri nedir?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

# Define character analysis step
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Bir film eleştirmenisin."),
            ("human", "Şu karakterleri analiz et: {characters}. Onları özel kılan nedir?"),
        ]
    )
    return character_template.format_prompt(characters=characters)


# Combine analyses into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# Simplify branches with LCEL
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

# Run the chain
result = chain.invoke({"movie_name": "aşk-ı memnu"})

print(result)

