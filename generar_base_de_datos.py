import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


AZURE_OPENAI_API_KEY            = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT           = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION        = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.environ["AZURE_OPENAI_EMBEDDING_MODEL_NAME"]
AZURE_EMBEDDING_DEPLOYMENT      = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
CHROMA_DB_DIR                   = os.environ["CHROMA_DB_DIR"]      # Carpeta donde Chroma guardará el índice
DATA_DIR_ABOGACIA                       = os.environ["DATA_DIR_ABOGACIA"]          # Carpeta donde tengas tus .txt
DATA_DIR_COMUNICACION                  = os.environ["DATA_DIR_COMUNICACION"]      # Carpeta donde tengas tus .txt
DATA_DIR_DISENO                = os.environ["DATA_DIR_DISENO"]    # Carpeta donde tengas tus .txt
DATA_DIR_ECONOMIA = os.environ["DATA_DIR_ECONOMIA"]  # Carpeta donde tengas tus .txt
DATA_DIR_ECONOMIA_EMPRESARIAL = os.environ["DATA_DIR_ECONOMIA_EMPRESARIAL"]  # Carpeta donde tengas tus .txt
DATA_DIR_FINANZAS = os.environ["DATA_DIR_FINANZAS"]  # Carpeta donde tengas tus .txt
DATA_DIR_HUMANIDADES = os.environ["DATA_DIR_HUMANIDADES"]  # Carpeta donde tengas tus .txt
DATA_DIR_INGENIERIA_EN_BIOTECNOLOGIA = os.environ["DATA_DIR_INGENIERIA_EN_BIOTECNOLOGIA"]  # Carpeta donde tengas tus .txt
DATA_DIR_INGENIERIA_EN_INTELIGENCIA_ARTIFICIAL = os.environ["DATA_DIR_INGENIERIA_EN_INTELIGENCIA_ARTIFICIAL"]  # Carpeta donde tengas tus .txt
DATA_DIR_INGENIERIA_EN_SUSTENTABILIDAD = os.environ["DATA_DIR_INGENIERIA_EN_SUSTENTABILIDAD"]  # Carpeta donde tengas tus .txt
DATA_DIR_ADMINISTRACION = os.environ["DATA_DIR_ADMINISTRACION"]  # Carpeta donde tengas tus .txt
DATA_DIR_CIENCIA_POLITICA_Y_GOBIERNO = os.environ["DATA_DIR_CIENCIA_POLITICA_Y_GOBIERNO"]  # Carpeta donde tengas tus .txt
DATA_DIR_CIENCIAS_DE_LA_EDUCACION = os.environ["DATA_DIR_CIENCIAS_DE_LA_EDUCACION"]  # Carpeta donde tengas tus .txt
DATA_DIR_CIENCIAS_DEL_COMPORTAMIENTO = os.environ["DATA_DIR_CIENCIAS_DEL_COMPORTAMIENTO"]  # Carpeta donde tengas tus .txt
DATA_DIR_NEGOCIOS_DIGITALES = os.environ["DATA_DIR_NEGOCIOS_DIGITALES"]  # Carpeta donde tengas tus .txt
DATA_DIR_RELACIONES_INTERNACIONALES = os.environ["DATA_DIR_RELACIONES_INTERNACIONALES"]  # Carpeta donde tengas tus .txt
DATA_DIR_PROFESORADO_EDUCACION_PRIMARIA = os.environ["DATA_DIR_PROFESORADO_EDUCACION_PRIMARIA"]  # Carpeta donde tengas tus .txt
DATA_DIR_PROGRAMAS_INTERNACIONALES = os.environ["DATA_DIR_PROGRAMAS_INTERNACIONALES"]  # Carpeta donde tengas tus .txt
DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_EXTRANJEROS = os.environ["DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_EXTRANJEROS"]  # Carpeta donde tengas tus .txt
DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_LOCALES = os.environ["DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_LOCALES"]  # Carpeta donde tengas tus .txt
DATA_DIR_INFO_CARRERAS_DE_GRADO = os.environ["DATA_DIR_INFO_CARRERAS_DE_GRADO"]  # Carpeta donde tengas tus .txt
DATA_DIR_CATEDRA_EEUU = os.environ["DATA_DIR_CATEDRA_EEUU"]  # Carpeta donde tengas tus .txt
DATA_DIR_BECAS_Y_ASISTENCIA_FINANCIERA = os.environ["DATA_DIR_BECAS_Y_ASISTENCIA_FINANCIERA"]  # Carpeta donde tengas tus .txt
DATA_DIR_DESARROLLO_PROFESIONAL =  os.environ["DATA_DIR_DESARROLLO_PROFESIONAL"]  # Carpeta donde tengas tus .txt



embedding_model = AzureOpenAIEmbeddings(
    azure_deployment    = AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_version  = AZURE_OPENAI_API_VERSION,
    azure_endpoint      = AZURE_OPENAI_ENDPOINT,
    openai_api_key      = AZURE_OPENAI_API_KEY,
)

def cargar_txt_como_documentos(directorio_txt, carrera):
    lista_documentos = []
    for nombre_archivo in os.listdir(directorio_txt):
        if not nombre_archivo.lower().endswith(".txt"):
            continue

        ruta = os.path.join(directorio_txt, nombre_archivo)
        loader = TextLoader(ruta, encoding="utf-8")
        docs_lineas = loader.load()
        texto_completo = "\n".join([d.page_content for d in docs_lineas])

        metadata = {
            "source": nombre_archivo,
            "carrera": carrera
        }
        lista_documentos.append(Document(page_content=texto_completo, metadata=metadata))

    return lista_documentos

def crear_vectorstore_txt(directorio_txt, persist_dir, carrera):
    documentos = cargar_txt_como_documentos(directorio_txt, carrera)
    if not documentos:
        print("No se encontraron archivos .txt en el directorio especificado.")
        return
    Chroma.from_documents(
        documents         = documentos,
        embedding         = embedding_model,
        persist_directory = persist_dir
    )
    print(f"[✔] Índice guardado en Chroma: '{persist_dir}' (carrera = {carrera})")

if __name__ == "__main__":
    crear_vectorstore_txt(DATA_DIR_ABOGACIA, CHROMA_DB_DIR, carrera="abogacia")
    crear_vectorstore_txt(DATA_DIR_COMUNICACION, CHROMA_DB_DIR, carrera="comunicacion")
    crear_vectorstore_txt(DATA_DIR_DISENO, CHROMA_DB_DIR, carrera="diseno")
    crear_vectorstore_txt(DATA_DIR_ECONOMIA, CHROMA_DB_DIR, carrera="economia")
    crear_vectorstore_txt(DATA_DIR_ECONOMIA_EMPRESARIAL, CHROMA_DB_DIR, carrera="economia_empresarial")
    crear_vectorstore_txt(DATA_DIR_FINANZAS, CHROMA_DB_DIR, carrera="finanzas")
    crear_vectorstore_txt(DATA_DIR_HUMANIDADES, CHROMA_DB_DIR, carrera="humanidades")
    crear_vectorstore_txt(DATA_DIR_INGENIERIA_EN_BIOTECNOLOGIA, CHROMA_DB_DIR, carrera="ingenieria_en_biotecnologia")
    crear_vectorstore_txt(DATA_DIR_INGENIERIA_EN_INTELIGENCIA_ARTIFICIAL, CHROMA_DB_DIR, carrera="ingenieria_en_inteligencia_artificial")
    crear_vectorstore_txt(DATA_DIR_INGENIERIA_EN_SUSTENTABILIDAD, CHROMA_DB_DIR, carrera="ingenieria_en_sustentabilidad")
    crear_vectorstore_txt(DATA_DIR_ADMINISTRACION, CHROMA_DB_DIR, carrera="administracion")
    crear_vectorstore_txt(DATA_DIR_CIENCIA_POLITICA_Y_GOBIERNO, CHROMA_DB_DIR, carrera="ciencia_politica_y_gobierno")
    crear_vectorstore_txt(DATA_DIR_CIENCIAS_DE_LA_EDUCACION, CHROMA_DB_DIR, carrera="ciencias_de_la_educacion")
    crear_vectorstore_txt(DATA_DIR_CIENCIAS_DEL_COMPORTAMIENTO, CHROMA_DB_DIR, carrera="ciencias_del_comportamiento")
    crear_vectorstore_txt(DATA_DIR_NEGOCIOS_DIGITALES, CHROMA_DB_DIR, carrera="negocios_digitales")
    crear_vectorstore_txt(DATA_DIR_RELACIONES_INTERNACIONALES, CHROMA_DB_DIR, carrera="relaciones_internacionales")
    crear_vectorstore_txt(DATA_DIR_PROFESORADO_EDUCACION_PRIMARIA, CHROMA_DB_DIR, carrera="profesorado_educacion_primaria")
    crear_vectorstore_txt(DATA_DIR_PROGRAMAS_INTERNACIONALES, CHROMA_DB_DIR, carrera="programas_internacionales")
    crear_vectorstore_txt(DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_EXTRANJEROS, CHROMA_DB_DIR, carrera="programas_internacionales_estudiantes_extranjeros")
    crear_vectorstore_txt(DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_LOCALES, CHROMA_DB_DIR, carrera="programas_internacionales_estudiantes_locales")
    crear_vectorstore_txt(DATA_DIR_INFO_CARRERAS_DE_GRADO, CHROMA_DB_DIR, carrera="info_carreras_de_grado")
    crear_vectorstore_txt(DATA_DIR_CATEDRA_EEUU, CHROMA_DB_DIR, carrera="catedra_eeuu")
    crear_vectorstore_txt(DATA_DIR_BECAS_Y_ASISTENCIA_FINANCIERA, CHROMA_DB_DIR, carrera="becas_y_asistencia_financiera")
    crear_vectorstore_txt(DATA_DIR_DESARROLLO_PROFESIONAL, CHROMA_DB_DIR, carrera="desarrollo_profesional")
    print("[✔] Base de datos creada exitosamente.")